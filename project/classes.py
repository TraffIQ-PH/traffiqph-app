from dataclasses import dataclass

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