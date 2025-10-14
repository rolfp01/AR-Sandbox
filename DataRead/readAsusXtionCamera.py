import numpy as np
import cv2
from primesense import openni2
from templates.base_camera_manager import BaseCameraManager
from templates.openNI import init_openni2

class AsusXtionCameraManager(BaseCameraManager):
    def __init__(self):
        self.device = None
        self.color_stream = None
        self.depth_stream = None

        # Attribute hinzufügen, damit der Zugriff funktioniert
        self.depth_scale = 0.001  # Beispiel: 1 mm = 0.001 m (kann angepasst werden)
        self.baseline_distance = None  # Wenn du es hast, sonst None
    
    def start(self):
        init_openni2()
        self.device = openni2.Device.open_any()
        if not self.device:
            raise RuntimeError("OpenNI2 Gerät konnte nicht geöffnet werden")

        self.color_stream = self.device.create_color_stream()
        self.color_stream.start()

        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.start()

        print("Asus XtionPRO Live erfolgreich initialisiert")

    def read_frame(self):
        # Nutze ausgelagerte Funktion
        return read_frames_asus(self.color_stream, self.depth_stream)

    def stop(self):
        if self.color_stream:
            self.color_stream.stop()
            self.color_stream = None
        if self.depth_stream:
            self.depth_stream.stop()
            self.depth_stream = None
        cv2.destroyAllWindows()
        print("Asus XtionPRO Live gestoppt")

def read_frames_asus(color_stream, depth_stream):
    """
    Liest Farbbild und Tiefenbild von Asus Xtion Kamera über OpenNI2.
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
        depth = np.frombuffer(
            d_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(
                (d_frame.height, d_frame.width))

        return {
            "color": color,
            "depth": depth
        }
    except Exception as e:
        print(f"[FEHLER] Fehler beim Lesen der Kamera-Frames: {e}")
        return None