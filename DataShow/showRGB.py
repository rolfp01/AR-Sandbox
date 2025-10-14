import cv2
from app import current_homography
import numpy as np

def show_Colors(calculationOutput):
    if calculationOutput is not None:
        if current_homography is not None:
            H = np.array(current_homography, dtype=np.float32)
            height, width = calculationOutput.shape[:2]
            calculationOutput = cv2.warpPerspective(calculationOutput, H, (width, height))

        if calculationOutput is None or calculationOutput.size == 0:
            print("[ERROR] Transformiertes Bild ist leer!")
            return None, None

        print(f"[DEBUG] Bildgröße nach Transformation: {calculationOutput.shape}")

        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', calculationOutput)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput