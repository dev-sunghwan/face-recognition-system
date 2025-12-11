# camera/frame_buffer.py

import threading

class FrameBuffer:
    def __init__(self, size=10):
        self.size = size
        self.buffer = []
        self.lock = threading.Lock()

    def add(self, frame):
        with self.lock:
            if len(self.buffer) >= self.size:
                self.buffer.pop(0)
            self.buffer.append(frame)

    def get_latest(self):
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]
