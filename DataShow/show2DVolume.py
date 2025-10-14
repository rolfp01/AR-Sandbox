import cv2
import numpy as np

def show_2D_Volume(calculationOutput,building_mask, road_mask, park_mask):
    if calculationOutput is not None:
        # Invertiere die Lärmkarte
        inverted_output = 255 - calculationOutput
        # Lärmkarte in Farbdarstellung (z.B. "HOT" colormap: gelb=ruhig, rot=laut)
        noise_colormap = cv2.applyColorMap(inverted_output, cv2.COLORMAP_AUTUMN)
        
        # Gebäude (pink, volle Deckkraft)
        if np.any(building_mask):
            noise_colormap[building_mask > 0] = (147,20,255)

        # Straßen (grau, volle Deckkraft)
        if np.any(road_mask):
            noise_colormap[road_mask > 0] = (50,50,50)

        # Parks (grün, volle Deckkraft)
        if np.any(park_mask):
            noise_colormap[park_mask > 0] = (0, 255, 0)
        
        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', noise_colormap)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput