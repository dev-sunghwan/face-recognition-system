# camera/rtsp_client.py

import socket
import re
import hashlib
import os
import threading


class RTSPClient:
    def __init__(self, url, verbose=True):
        self.url = url
        self.verbose = verbose
        self.cseq = 1
        self.session_id = None
        self.socket = None

        self.realm = None
        self.nonce = None
        self.qop = None
        self.cnonce = None
        self.nc = 1

        self.video_track_uri = None
        self.server_port_rtp = None
        self.rtp_socket = None

    def log(self, *args):
        if self.verbose:
            print("[RTSP]", *args)

    # -------------------------------------------
    # URL parsing & socket connect
    # -------------------------------------------
    def connect(self):
        m = re.match(r"rtsp://(.*?):(.*?)@(.*?)/(.*)", self.url)
        if not m:
            raise ValueError("Invalid RTSP URL")

        self.username = m.group(1)
        self.password = m.group(2)
        self.host = m.group(3)
        self.stream_path = m.group(4)

        self.log("Connecting to", self.host)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, 554))

    # -------------------------------------------
    # Digest Calculation
    # -------------------------------------------
    def _digest_header(self, method, uri):
        HA1 = hashlib.md5(f"{self.username}:{self.realm}:{self.password}".encode()).hexdigest()
        HA2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()

        if not self.cnonce:
            self.cnonce = hashlib.md5(os.urandom(16)).hexdigest()

        nc_value = f"{self.nc:08x}"

        response = hashlib.md5(
            f"{HA1}:{self.nonce}:{nc_value}:{self.cnonce}:{self.qop}:{HA2}".encode()
        ).hexdigest()

        self.nc += 1

        header = (
            f'Authorization: Digest username="{self.username}", '
            f'realm="{self.realm}", '
            f'nonce="{self.nonce}", '
            f'uri="{uri}", '
            f'response="{response}", '
            f'qop={self.qop}, '
            f'nc={nc_value}, '
            f'cnonce="{self.cnonce}"\r\n'
        )
        return header

    # -------------------------------------------
    # Send RTSP command
    # -------------------------------------------
    def _send_rtsp(self, method, uri, extra_headers=""):
        req = f"{method} rtsp://{self.host}{uri} RTSP/1.0\r\n"
        req += f"CSeq: {self.cseq}\r\n"

        if self.realm and self.nonce:
            req += self._digest_header(method, uri)

        req += extra_headers
        req += "\r\n"

        self._send(req)
        resp = self._recv()

        # Handle 401 → extract Digest auth info → retry
        if "401 Unauthorized" in resp:
            self._parse_digest(resp)
            req = f"{method} rtsp://{self.host}{uri} RTSP/1.0\r\n"
            req += f"CSeq: {self.cseq}\r\n"
            req += self._digest_header(method, uri)
            req += extra_headers + "\r\n"
            self._send(req)
            resp = self._recv()

        self.cseq += 1
        return resp

    def _send(self, data):
        self.socket.send(data.encode())

    def _recv(self):
        return self.socket.recv(5000).decode()

    # -------------------------------------------
    # Parse Digest Auth
    # -------------------------------------------
    def _parse_digest(self, resp):
        def extract(field):
            m = re.search(field + r'="(.*?)"', resp)
            return m.group(1) if m else None

        self.realm = extract("realm")
        self.nonce = extract("nonce")
        self.qop = extract("qop")

        self.log("Digest realm:", self.realm)
        self.log("Digest nonce:", self.nonce)
        self.log("Digest qop:", self.qop)

    # -------------------------------------------
    # RTSP commands
    # -------------------------------------------
    def options(self):
        return self._send_rtsp("OPTIONS", f"/{self.stream_path}")

    def describe(self):
        resp = self._send_rtsp(
            "DESCRIBE", f"/{self.stream_path}", "Accept: application/sdp\r\n"
        )

        print("\n===== DESCRIBE RAW RESPONSE =====")
        print(resp)
        print("================================\n")

        # Extract video track URI
        m = re.search(r'a=control:(rtsp://\S+trackID=v)', resp)
        if m:
            full_uri = m.group(1)
            # Convert full RTSP URI into path-only
            self.video_track_uri = full_uri.split(self.host)[1]
            self.log("Video Track URI:", self.video_track_uri)
        else:
            raise RuntimeError("Video track (trackID=v) not found in DESCRIBE.")

        return resp

    def setup(self):
        if not self.video_track_uri:
            raise RuntimeError("DESCRIBE must run before SETUP.")

        resp = self._send_rtsp(
            "SETUP",
            self.video_track_uri,
            "Transport: RTP/AVP;unicast;client_port=8000-8001\r\n"
        )

        print("\n===== SETUP RAW RESPONSE =====")
        print(resp)
        print("================================\n")

        # Session
        m = re.search(r"Session: (\S+)", resp)
        if m:
            self.session_id = m.group(1)
            self.log("Session:", self.session_id)

        # RTP server port
        m = re.search(r"server_port=(\d+)-", resp)
        if m:
            self.server_port_rtp = int(m.group(1))
        else:
            self.server_port_rtp = 8000

        self.log("RTP server port:", self.server_port_rtp)
        return resp

    def play(self):
        response = self._send_rtsp(
            "PLAY",
            self.video_track_uri,
            f"Session: {self.session_id}\r\n"
        )
    
        # Keep-alive Thread start
        threading.Thread(target=self.keep_alive, daemon=True).start()

        return response
    

    # -------------------------------------------
    # RTP Socket
    # -------------------------------------------
    def open_rtp_socket(self):
        self.log("Opening RTP socket on port 8000")
        self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp_socket.bind(("", 8000))
        self.rtp_socket.settimeout(1)

    def receive_rtp_packet(self):
        try:
            packet, addr = self.rtp_socket.recvfrom(4096)
            return packet
        except socket.timeout:
            return None


    def keep_alive(self):
        """
        Periodically send GET_PARAMETER to prevent RTSP session timeout.
        """
        import time
        
        while True:
            # keep-alive every 30 seconds
            time.sleep(30)

            if not self.session_id:
                continue

            msg = (
                f"GET_PARAMETER {self.url} RTSP/1.0\r\n"
                f"CSeq: {self.cseq}\r\n"
                f"Session: {self.session_id}\r\n"
                f"\r\n"
            )

            try:
                self.socket.send(msg.encode("utf-8"))
                print(f"[RTSP] Sent keep-alive GET_PARAMETER (CSeq {self.cseq})")
                self.cseq += 1
            except Exception as e:
                print("[RTSP] Keep-alive failed:", e)
                break


if __name__ == "__main__":
    client = RTSPClient("rtsp://user:pass@192.168.1.100/profile2")
    client.connect()
    print(client.options())
    print(client.describe())
    print(client.setup())

    
