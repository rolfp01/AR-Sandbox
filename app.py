from flask import Flask, redirect, render_template, Response, session, url_for, jsonify

from DataCalculation import calculate2DVolume, calculateHight, detectBuildings, grayPicture
from DataRead import readAsusXtionCamera, readLaptopCamera, readIntelD415Camera
from DataShow import show2DVolume, showGrayPicture, showHight, showObjects, showColorAndDepth
import numpy as np
import cv2
from openni import openni2
import pyrealsense2 as rs

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Füge hier deinen geheimen Schlüssel hinzu

# Definition der VideoTheme-Klasse
class VideoTheme:
    def __init__(self, index, name, funct):
        self.index = index
        self.name = name
        self.funct = funct

def gray_video():
    try:
        camera = cv2.VideoCapture(0) # 0 für die Standardkamera
        while True:
            success, frame = camera.read() # Bild von der Kamera lesen
            if success:
                cameraInput = readLaptopCamera.read_Laptop_Camera(frame)
                if type(cameraInput) == 'NoneType':
                    continue
                else: 
                    calculationOutput = grayPicture.picture_In_Gray(cameraInput)
                    beamerOutput = showGrayPicture.show_Gray_Picture(calculationOutput)
                yield beamerOutput
        
    finally:
        openni2.unload()
        cv2.destroyAllWindows()
        
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

def objects_video():
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

    # Maßstab für Tiefenwerte ermitteln (wie viele Meter pro Tiefen-Einheit)
    ###depth_sensor = profile.get_device().first_depth_sensor()
    ###depth_scale = depth_sensor.get_depth_scale()
    ###print(f"Tiefen-Skalierungsfaktor: {depth_scale} m pro Einheit")
    depth_scale = 1

    # Optional: Grundabstand kalibrieren (Tiefe zum Sandkastenboden)
    # Man kann z.B. bei leerer Sandbox die gemessene Tiefe des Bodens erfassen.
    # Hier nehmen wir an, baseline_distance ist unbekannt und verwenden relative Höhe.
    baseline_distance = None  # in Metern (falls bekannt, z.B. gemessen)

    try:
        while True:
            color_and_depth_image = readAsusXtionCamera.read_Depth_Camera(color_stream, depth_stream)
            building_mask, road_mask, park_mask = detectBuildings.detect_Buildings(color_and_depth_image["depth"], color_and_depth_image["color"], depth_scale, baseline_distance)
            ret, beamerOutput = showObjects.show_Objects(building_mask, road_mask, park_mask)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput
    finally:
        # Stoppe die Streams und schließe alle Fenster
        color_stream.stop()
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

def volume_2D_video():
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

    # Maßstab für Tiefenwerte ermitteln (wie viele Meter pro Tiefen-Einheit)
    ###depth_sensor = profile.get_device().first_depth_sensor()
    ###depth_scale = depth_sensor.get_depth_scale()
    ###print(f"Tiefen-Skalierungsfaktor: {depth_scale} m pro Einheit")
    depth_scale = 1

    # Optional: Grundabstand kalibrieren (Tiefe zum Sandkastenboden)
    # Man kann z.B. bei leerer Sandbox die gemessene Tiefe des Bodens erfassen.
    # Hier nehmen wir an, baseline_distance ist unbekannt und verwenden relative Höhe.
    baseline_distance = None  # in Metern (falls bekannt, z.B. gemessen)

    try:
        while True:
            color_and_depth_image = readAsusXtionCamera.read_Depth_Camera(color_stream, depth_stream)
            building_mask, road_mask, park_mask = detectBuildings.detect_Buildings(color_and_depth_image["depth"], color_and_depth_image["color"], depth_scale, baseline_distance)
            calculationOutput = calculate2DVolume.calculate_2D_Volume(color_and_depth_image['depth'], building_mask, road_mask, park_mask)
            ret, beamerOutput = show2DVolume.show_2D_Volume(calculationOutput)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput
    finally:
        # Stoppe die Streams und schließe alle Fenster
        color_stream.stop()
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        
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

def intel_video():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            color_and_depth_image = readIntelD415Camera.read_Depth_Camera(pipeline)
            ret, beamerOutput = showColorAndDepth.show_Color_And_Depth(cv2.convertScaleAbs(color_and_depth_image["depth"], alpha=0.03), color_and_depth_image["color"])
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput

    finally:
        # Ressourcen freigeben
        pipeline.stop()
        cv2.destroyAllWindows()

activeVideoTheme = 0
videoThemes = [
    VideoTheme(0, "Graustufen Video", gray_video),
    VideoTheme(1, "Objekte", objects_video),
    VideoTheme(2, "2D Lautstärke TODO", color_video),
    VideoTheme(3, "Höhe", hights_video),
    VideoTheme(4, "Intel Real Sense", intel_video)

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
