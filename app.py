from flask import Flask, redirect, render_template, Response, session, url_for, jsonify

from DataCalculation import calculate2DVolume, calculateHight, detectBuildings, grayPicture
from DataRead import readAsusXtionCamera, readLaptopCamera
from DataShow import show2DVolume, showGrayPicture, showHight, showObjects
import numpy as np
import cv2
from openni import openni2

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Füge hier deinen geheimen Schlüssel hinzu

# Definition der VideoTheme-Klasse
class VideoTheme:
    def __init__(self, index, name, funct):
        self.index = index
        self.name = name
        self.funct = funct

def gray_video():
    while True:
        cameraInput = readLaptopCamera.read_Laptop_Camera()
        if type(cameraInput) == 'NoneType':
            continue
        else: 
            calculationOutput = grayPicture.picture_In_Gray(cameraInput)
            beamerOutput = showGrayPicture.show_Gray_Picture(calculationOutput)
        yield beamerOutput

def color_video():
        # Initialisiere OpenNI2
        openni2.initialize()

        # Öffne die Kamera
        dev = openni2.Device.open_any()

        # Öffne den Farbsensor und den Tiefensensor
        color_stream = dev.create_color_stream()

        # Starte beide Streams
        color_stream.start()

        try:
            while True:

                # Lese ein Frame vom Farbsensor
                color_frame = color_stream.read_frame()
                color_data = color_frame.get_buffer_as_uint8()
                color_image = np.frombuffer(color_data, dtype=np.uint8).reshape((color_frame.height, color_frame.width, 3))

                # Wenn das Bild im RGB-Format vorliegt (was häufig der Fall ist), invertiere die Kanäle:
                color_image = color_image[..., ::-1]  # BGR -> RGB Umkehrung

                # Konvertiere das Bild in JPEG-Format
                ret, buffer = cv2.imencode('.jpg', color_image)
                frame = buffer.tobytes() # Bild in Bytes umwandeln
                beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                # Wenn 'q' gedrückt wird, beenden
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                yield beamerOutput

        finally:
            # Stoppe den Farbsensor und schließe OpenNI2
            color_stream.stop()
            openni2.unload()
            cv2.destroyAllWindows()

'''
def depth_video():
    # Initialisiere OpenNI2
    openni2.initialize()

    # Öffne die Kamera
    dev = openni2.Device.open_any()

    # Öffne den Farbsensor und den Tiefensensor
    depth_stream = dev.create_depth_stream()

    # Starte beide Streams
    depth_stream.start()

    try:
        while True:

            # Lese ein Frame vom Tiefensensor
            depth_frame = depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_frame.height, depth_frame.width))

            # Normalisiere das Tiefenbild (für bessere Darstellung)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)

            # Tiefenbild einfärben
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
            # Konvertiere das Bild in JPEG-Format
            ret, buffer = cv2.imencode('.jpg', depth_colormap)
            frame = buffer.tobytes() # Bild in Bytes umwandeln
            beamerOutput = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Wenn 'q' gedrückt wird, beenden
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield beamerOutput

    finally:
        # Stoppe den Farbsensor und schließe OpenNI2
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
'''

'''
def objects_video():
    #while True:
        cameraInput = readDepthCamera.read_Depth_Camera()
        #calculationOutput = detectBuildings.detect_Buildings(cameraInput)
        #beamerOutput = showObjects.show_Objects(calculationOutput)
        #yield beamerOutput

def volume_2D_video():
    #while True:
        cameraInput = readDepthCamera.read_Depth_Camera()
        #calculationOutput = calculate2DVolume.calculate_2D_Volume(cameraInput)
        #beamerOutput = show2DVolume.show_2D_Volume(calculationOutput)
        #yield beamerOutput
        '''

def hights_video():
    # Initialisiere OpenNI2
    openni2.initialize()

    # Öffne die Kamera
    dev = openni2.Device.open_any()

    # Öffne den Farbsensor und den Tiefensensor
    color_stream = dev.create_color_stream()
    depth_stream = dev.create_depth_stream()

    # Starte beide Streams
    color_stream.start()
    depth_stream.start()
    try:
        while True:
            color_and_depth_image = readAsusXtionCamera.read_Depth_Camera(color_stream, depth_stream)
            calculationOutput = calculateHight.calculate_Hight(color_and_depth_image["depth"])
            ret, beamerOutput = showHight.show_Hights(calculationOutput)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput
    finally:
        # Stoppe die Streams und schließe alle Fenster
        color_stream.stop()
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

activeVideoTheme = 0
videoThemes = [
    VideoTheme(0, "Graustufen Video", gray_video),
    VideoTheme(1, "Objekte", color_video),
    VideoTheme(2, "2D Lautstärke", color_video),
    VideoTheme(3, "Höhe", hights_video)
]

@app.route('/')
def index():    # Überprüfen, ob 'videoTheme' in der Session existiert
    if 'activeVideoTheme' not in session:
        session['activeVideoTheme'] = 0  # Wenn nicht, auf das erste Thema setzen

    current_theme = videoThemes[session['activeVideoTheme']].name  # Hole den aktuellen Themennamen
    return render_template('index.html', current_theme=current_theme)  # Übergib den Namen an das Template

@app.route('/video_feed')
def video_feed():
    return Response(videoThemes[session['activeVideoTheme']].funct(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/theme_Switch')
def theme_Switch():
    print("test")
    # Wechsle das Thema
    session['activeVideoTheme'] = (session['activeVideoTheme'] + 1) % len(videoThemes)  # Einfacher Wechsel der Themenindex
    print(f"Thema gewechselt zu: {videoThemes[session['activeVideoTheme']].name}")
    return redirect(url_for('index'))  # Redirect zur Startseite, um das Template mit dem neuen Thema zu laden

if __name__ == '__main__':
    app.run(debug=True)
