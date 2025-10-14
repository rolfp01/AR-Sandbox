import numpy as np
import cv2
from templates.base_camera_manager import BaseCameraManager

class KinectCameraManager(BaseCameraManager):
    """Manager für Microsoft Kinect Kamera (Kinect v2) geht nur mit Python 3.8"""
    
    def __init__(self):
        super().__init__()
        self.kinect = None
    
    def start(self):
        from pykinect2 import PyKinectRuntime, PyKinectV2
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        print("Microsoft Kinect erfolgreich initialisiert")
    
    def read_frame(self):
        if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():
            color_frame = self.kinect.get_last_color_frame()
            depth_frame = self.kinect.get_last_depth_frame()
            
            # Color Frame in ein numpy Array umwandeln (BGRA 1920x1080)
            color_image = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
            # In BGR konvertieren für OpenCV (falls gewünscht)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            
            # Depth Frame (512x424) in numpy Array
            depth_image = depth_frame.reshape((424, 512)).astype(np.uint16)
            
            return {
                "color": color_image,
                "depth": depth_image
            }
        return None
    
    def stop(self):
        if self.kinect:
            self.kinect.close()
        cv2.destroyAllWindows()
        print("Microsoft Kinect gestoppt")
