import numpy as np
import cv2
from openni import openni2

def calculate_Hight(depth_image):
    if depth_image is None:
        raise ValueError("depth_image is None, please check your depth camera input")
    #if depth_image.dtype != np.uint8:
    #    depth_image = np.uint8(depth_image)

    # Bestimme den minimalen und maximalen Tiefenwert
    min_depth = np.min(depth_image)
    max_depth = np.max(depth_image)

    # Verhindere die Division durch Null, wenn alle Werte gleich sind
    if min_depth == max_depth:
        raise ValueError("The depth image has no variation in depth values.")

    # Normalisiere das Tiefenbild (für bessere Darstellung)
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)

    # Tiefenbild einfärben
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap