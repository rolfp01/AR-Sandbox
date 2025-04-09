import numpy as np
import cv2
from openni import openni2

def show_Hights(calculationOutput):
    if calculationOutput is not None:
        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', calculationOutput)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput