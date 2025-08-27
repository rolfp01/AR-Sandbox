import cv2

def show_Colors(calculationOutput):
    if calculationOutput is not None:
        # Konvertiere das Bild in JPEG-Format
        ret, buffer = cv2.imencode('.jpg', calculationOutput)
        frame = buffer.tobytes() # Bild in Bytes umwandeln
        beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return ret, beamerOutput