"""
Network Server Module — PROTOCOL FIXED
=======================================
Frame receive : 4-byte BIG-ENDIAN uint   + JPEG bytes   ← matches Unity Array.Reverse
Result send   : 4-byte LITTLE-ENDIAN int + UTF-8 JSON   ← matches Unity BitConverter.ToInt32
"""
import socket
import struct
import json


class NetworkServer:
    def __init__(self, ip, port, logger):
        self.ip     = ip
        self.port   = port
        self.logger = logger
        self.server_socket = None
        self.connection    = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(1)
        self.logger.info(f"[NETWORK] Listening on {self.ip}:{self.port}")
        self.logger.info("=" * 60)
        self.logger.info("READY - waiting for mobile connection...")
        self.logger.info("=" * 60)
        conn, addr = self.server_socket.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection = conn
        self.logger.info(f"[NETWORK] Connected: {addr}")
        return conn

    def _recv_exact(self, conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def receive_frame(self, conn):
        """
        Unity sends:
          4 bytes  BIG-ENDIAN unsigned int  (Array.Reverse on little-endian PC)
          N bytes  JPEG data
        """
        try:
            size_data = self._recv_exact(conn, 4)
            if size_data is None:
                return None
            frame_size = struct.unpack(">I", size_data)[0]   # ">I" = big-endian
            if frame_size == 0 or frame_size > 10_000_000:
                self.logger.error(f"[NETWORK] Bad frame size: {frame_size}")
                return None
            return self._recv_exact(conn, frame_size)
        except Exception as e:
            self.logger.error(f"[NETWORK] Receive error: {e}")
            return None

    def send_result(self, conn, result):
        """
        Unity expects:
          4 bytes  LITTLE-ENDIAN signed int  (BitConverter.ToInt32)
          N bytes  UTF-8 JSON
        """
        try:
            json_bytes    = json.dumps(result).encode("utf-8")
            length_prefix = struct.pack("<i", len(json_bytes))   # "<i" = little-endian
            conn.sendall(length_prefix + json_bytes)
            return True
        except Exception as e:
            self.logger.error(f"[NETWORK] Send error: {e}")
            return False

    def cleanup(self):
        for s in (self.connection, self.server_socket):
            if s:
                try: s.close()
                except: pass
        self.connection    = None
        self.server_socket = None