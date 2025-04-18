import cv2

def show_2D_Volume(calculationOutput):
    if calculationOutput is not None:
        # Lärmkarte in Farbdarstellung (z.B. "HOT" colormap: schwarz=ruhig, rot/gelb=laut)
        noise_colormap = cv2.applyColorMap(calculationOutput, cv2.COLORMAP_HOT)
        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', noise_colormap)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput