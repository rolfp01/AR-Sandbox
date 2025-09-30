import numpy as np
import cv2

def read_frames(color_stream, depth_stream):
    """
    Liest Farbbild und Tiefenbild von Asus Xtion Kamera über OpenNI2.
    Gibt dict mit 'color' und 'depth' zurück oder None bei Fehler.
    """
    try:
        # Farb-Frame lesen
        c_frame = color_stream.read_frame()
        width, height = c_frame.width, c_frame.height
        c_data = c_frame.get_buffer_as_uint8()

        if len(c_data) == width * height * 3:
            color = np.frombuffer(c_data, dtype=np.uint8).reshape((height, width, 3))
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        elif len(c_data) == width * height:
            color = np.frombuffer(c_data, dtype=np.uint8).reshape((height, width))
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        else:
            print(f"[WARNUNG] Unerwartete Farbbildgröße: {len(c_data)} Bytes")
            return None

        # Tiefenbild lesen
        d_frame = depth_stream.read_frame()
        depth = np.frombuffer(d_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((d_frame.height, d_frame.width))

        return {
            "color": color,
            "depth": depth
        }
    except Exception as e:
        print(f"[FEHLER] Fehler beim Lesen der Kamera-Frames: {e}")
        return None