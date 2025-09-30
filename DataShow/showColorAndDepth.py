import numpy as np
import cv2

def show_Color_And_Depth(depth_image, color_image):
    if depth_image is not None and color_image is not None:
        # Tiefenbild einfärben
        depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_RAINBOW)
        # RGB- und Tiefenbild nebeneinander anzeigen
        images = np.hstack((color_image, depth_colormap))

        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', images)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput