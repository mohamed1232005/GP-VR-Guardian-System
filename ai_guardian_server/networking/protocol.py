# ===========================================================================
# networking/protocol.py — Packet parsing utilities
# Parses the Unity frame packet format:
#   [4 bytes: metadata JSON length (little-endian)]
#   [N bytes: metadata JSON (UTF-8)]
#   [M bytes: JPEG image data]
# ===========================================================================

import struct
import json


def parse_frame_packet(data: bytes) -> tuple:
    """
    Parse a raw UDP packet from Unity into (metadata_dict, jpeg_bytes).
    
    Returns:
        (metadata: dict, jpeg_bytes: bytes) on success
        (None, None) on failure
    """
    if len(data) < 4:
        return None, None

    # First 4 bytes = little-endian int32 = metadata JSON length
    json_length = struct.unpack("<I", data[:4])[0]

    if len(data) < 4 + json_length:
        return None, None

    # Parse metadata JSON
    try:
        json_bytes = data[4:4 + json_length]
        metadata = json.loads(json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[P2_PROTOCOL] JSON parse error: {e}")
        return None, None

    # Remaining bytes = JPEG data
    jpeg_bytes = data[4 + json_length:]

    return metadata, jpeg_bytes
