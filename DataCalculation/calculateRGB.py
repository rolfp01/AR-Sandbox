import numpy as np
import cv2

def calculate_Colors(color_image):
    # Wenn das Bild im RGB-Format vorliegt (was häufig der Fall ist), invertiere die Kanäle:
    color_image = color_image[..., ::-1]  # BGR -> RGB Umkehrung
    return color_image