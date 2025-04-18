import numpy as np
import cv2
from openni import openni2

def show_Hights(calculationOutput):
    if calculationOutput is not None:
        # Tiefenbild einfärben
        depth_colormap = cv2.applyColorMap(calculationOutput, cv2.COLORMAP_JET)
        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', depth_colormap)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput