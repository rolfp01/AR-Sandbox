import numpy as np
import cv2

def show_Objects(building_mask, road_mask, park_mask, color_image):
    # Bildgröße von einer der Masken ableiten
    height, width = building_mask.shape

    # Erstelle ein transparentes RGBA-Bild
    output_img = np.zeros((height, width, 4), dtype=np.uint8)

    # Gebäude (rot, volle Deckkraft)
    if np.any(building_mask):
        output_img[building_mask > 0] = (0, 0, 255, 255)  # BGR + Alpha

    # Straßen (grau, volle Deckkraft)
    if np.any(road_mask):
        output_img[road_mask > 0] = (50,50,50, 255)

    # Parks (grün, volle Deckkraft)
    if np.any(park_mask):
        output_img[park_mask > 0] = (0, 255, 0, 255)
        
    color_rgba = cv2.cvtColor(color_image, cv2.COLOR_BGR2BGRA)
    color_rgba[:, :, 3] = 255  # Volle Deckkraft

    # RGB- und Tiefenbild nebeneinander anzeigen
    images = np.hstack((output_img, color_rgba))

    # Als PNG mit Alphakanal encodieren
    ret, buffer = cv2.imencode('.png', images)
    frame = buffer.tobytes() # Bild in Bytes umwandeln
    beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return ret, beamerOutput
