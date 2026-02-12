"""
Network Server Module
Handles TCP connection, frame receiving, and result sending
"""

import socket
import struct
import json


class NetworkServer:
    def __init__(self, ip, port, logger):
        """Initialize network server"""
        self.ip = ip
        self.port = port
        self.logger = logger
        self.server_socket = None
        self.connection = None
    
    def start(self):
        """Start server and wait for connection"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(1)
            
            self.logger.info(f"[NETWORK] ✓ Listening on {self.ip}:{self.port}")
            self.logger.info("\n" + "=" * 70)
            self.logger.info("READY! Waiting for mobile connection...")
            self.logger.info("=" * 70 + "\n")
            
            conn, addr = self.server_socket.accept()
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            self.connection = conn
            self.logger.info(f"[NETWORK] ✓✓✓ Connected: {addr}")
            
            return conn
        
        except Exception as e:
            self.logger.error(f"[NETWORK] Failed to start: {e}")
            raise
    
    def receive_frame(self, conn):
        """Receive JPEG frame from Unity"""
        try:
            # Read frame size (4 bytes, big-endian)
            size_data = b""
            while len(size_data) < 4:
                packet = conn.recv(4 - len(size_data))
                if not packet:
                    return None
                size_data += packet
            
            frame_size = struct.unpack(">L", size_data)[0]
            
            # Read frame data
            frame_data = b""
            while len(frame_data) < frame_size:
                packet = conn.recv(min(65536, frame_size - len(frame_data)))
                if not packet:
                    return None
                frame_data += packet
            
            return frame_data
        
        except Exception as e:
            self.logger.error(f"[NETWORK] Receive error: {e}")
            return None
    
    def send_result(self, conn, result):
        """Send JSON result to Unity"""
        try:
            json_str = json.dumps(result) + "\n"
            json_bytes = json_str.encode('utf-8')
            conn.sendall(json_bytes)
            return True
        
        except Exception as e:
            self.logger.error(f"[NETWORK] Send error: {e}")
            return False
    
    def cleanup(self):
        """Close connections"""
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass