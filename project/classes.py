from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class Camera:
    name: str
    ip_address: str
    username: str
    password: str
    rtsp_port: int
    channel: int
    location: str
    
    def __post_init__(self):
        self.full_link = f'rtsp://{self.username}:{self.password}@{self.ip_address}:{self.rtsp_port}/Streaming/Channels/{self.channel}'

    def __str__(self):
        return f"Camera: {self.name} at {self.full_link}"


class LatencyTracker:
    def __init__(self, window=300):
        self.capture = deque(maxlen=window)
        self.inference = deque(maxlen=window)
        self.display = deque(maxlen=window)

    def record_capture(self, t): self.capture.append(t)
    def record_inference(self, t): self.inference.append(t)
    def record_display(self, t): self.display.append(t)

    def summary(self):
        def avg(q): return np.mean(q) if q else 0
        return {
            "capture": avg(self.capture),
            "inference": avg(self.inference),
            "display": avg(self.display),
            "total": avg(self.capture) + avg(self.inference) + avg(self.display)
        }
