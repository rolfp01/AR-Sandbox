import cv2
from templates.base_camera_manager import BaseCameraManager

class LaptopCameraManager(BaseCameraManager):
    """Manager für Laptop-Kamera (Webcam)"""
    
    def __init__(self):
        super().__init__()
        self.camera = None
    
    def start(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Laptop-Kamera erfolgreich initialisiert")
    
    def read_frame(self):
        success, frame = self.camera.read()
        if success:
            color_image = read_Laptop_Camera(frame)
            # Rückgabe im gleichen Format wie andere Kameras
            return {
                "color": color_image,
                "depth": None  # Laptop-Kamera hat keine Tiefendaten
            }
        return None
    
    def stop(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Laptop-Kamera gestoppt")

def read_Laptop_Camera(frame):
    cameraInput = frame
    return cameraInput