# camera/decoder.py

import av
import numpy as np

class H264Decoder:
    def __init__(self):
        self.codec = av.CodecContext.create("h264", "r")

    def decode(self, nal_unit):
        """Decode a raw H.264 NAL unit into frames."""
        frames = []
        packets = self.codec.parse(nal_unit)
        for p in packets:
            f_list = self.codec.decode(p)
            frames.extend(f_list)
        return frames
