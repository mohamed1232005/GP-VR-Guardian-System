# vr_bridge.py
import socket
import json
import time

class VRBridge:
    def __init__(self, host="127.0.0.1", port=5555):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, landmarks, feedback):
        # Package landmarks and feedback into JSON
        data = {
            "landmarks": landmarks,
            "feedback": feedback,
            "timestamp": time.time()
        }
        message = json.dumps(data).encode('utf-8')
        self.sock.sendto(message, (self.host, self.port))

    def close(self):
        self.sock.close()
