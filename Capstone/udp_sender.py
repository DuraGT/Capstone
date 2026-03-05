"""
udp_sender.py
-------------
Thin wrapper around a non-blocking UDP socket.
Sends "X,Y,Z,Roll,Pitch,Yaw" strings to a remote host.
"""

import socket


class UDPSender:
    def __init__(self, target_ip: str, target_port: int):
        self.target_ip   = target_ip
        self.target_port = target_port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.connect((target_ip, target_port))
        print(f"  UDP → {target_ip}:{target_port}")

    def send(self, pose) -> bool:
        """Send pose as comma-separated string. Returns True on success."""
        try:
            msg = (f"{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f},"
                   f"{pose[3]:.2f},{pose[4]:.2f},{pose[5]:.2f}").encode()
            self._sock.send(msg)
            return True
        except (BlockingIOError, OSError):
            return False

    def close(self):
        self._sock.close()