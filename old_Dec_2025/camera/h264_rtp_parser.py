# camera/h264_rtp_parser.py

# H.264 RTP payload parser with FU-A reassembly
class H264RTPParser:
    def __init__(self):
        self.buffer = bytearray()  # For rebuilding fragmented NAL
        self.started = False       # Whether we are in a FU-A sequence

    def feed(self, packet):
        """
        Feed raw RTP packet bytes and return a complete NAL unit (or None)
        """
        if len(packet) < 13:
            return None

        # RTP HEADER (12 bytes)
        rtp_header = packet[:12]
        payload = packet[12:]

        # H264 NAL header (1 byte)
        nal_header = payload[0]
        nal_type = nal_header & 0x1F

        # ---- Case 1: Single NALU ----
        if nal_type > 0 and nal_type < 24:
            # This is a complete NAL unit (SPS, PPS, IDR, etc.)
            return b"\x00\x00\x00\x01" + payload

        # ---- Case 2: FU-A Fragmentation ----
        if nal_type == 28:  # FU-A
            fu_indicator = nal_header
            fu_header = payload[1]
            start_bit = fu_header >> 7
            end_bit = (fu_header >> 6) & 0x01
            fu_type = fu_header & 0x1F

            nal_unit_header = (fu_indicator & 0xE0) | fu_type

            # Start of fragmented NAL
            if start_bit == 1:
                self.buffer = bytearray()
                self.buffer.extend(b"\x00\x00\x00\x01")
                self.buffer.append(nal_unit_header)
                self.buffer.extend(payload[2:])
                self.started = True
                return None

            # Middle fragment
            if self.started and start_bit == 0 and end_bit == 0:
                self.buffer.extend(payload[2:])
                return None

            # End fragment
            if self.started and end_bit == 1:
                self.buffer.extend(payload[2:])
                self.started = False
                return bytes(self.buffer)

        # Other NAL types ignored
        return None


