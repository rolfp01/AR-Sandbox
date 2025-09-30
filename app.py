from flask import Flask, redirect, render_template, Response, session, url_for, jsonify

from DataCalculation import calculate2DVolume, calculateHight, calculateRGB, detectBuildings, grayPicture
from DataRead import readAsusXtionCamera, readLaptopCamera, readIntelD415Camera
from DataShow import show2DVolume, showGrayPicture, showHight, showRGB, showObjects, showColorAndDepth
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
            calculationOutput = grayPicture.picture_In_Gray(color_and_depth_image["color"])
            beamerOutput = showGrayPicture.show_Gray_Picture(calculationOutput)
            
            yield beamerOutput

    finally:
        # Ressourcen freigeben
        pipeline.stop()
        cv2.destroyAllWindows()
        
def color_video():    
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
            calculationOutput = calculateRGB.calculate_Colors(color_and_depth_image["color"])
            ret, beamerOutput = showRGB.show_Colors(calculationOutput)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput

    finally:
        # Ressourcen freigeben
        pipeline.stop()
        cv2.destroyAllWindows()

def objects_video():
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
            color_and_depth_image = readIntelD415Camera.read_Depth_Camera(pipeline)
            building_mask, road_mask, park_mask = detectBuildings.detect_Buildings(color_and_depth_image["depth"], color_and_depth_image["color"], depth_scale, baseline_distance)
            ret, beamerOutput = showObjects.show_Objects(building_mask, road_mask, park_mask)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput
    finally:
        # Ressourcen freigeben
        pipeline.stop()
        cv2.destroyAllWindows()

def volume_2D_video():
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
            color_and_depth_image = readIntelD415Camera.read_Depth_Camera(pipeline)
            building_mask, road_mask, park_mask = detectBuildings.detect_Buildings(color_and_depth_image["depth"], color_and_depth_image["color"], depth_scale, baseline_distance)
            calculationOutput = calculate2DVolume.calculate_2D_Volume(color_and_depth_image['depth'], building_mask, road_mask, park_mask)
            ret, beamerOutput = show2DVolume.show_2D_Volume(calculationOutput)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput
    finally:
        # Ressourcen freigeben
        pipeline.stop()
        cv2.destroyAllWindows()
        
def hights_video():
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
            calculationOutput = calculateHight.calculate_Hight(color_and_depth_image["depth"])
            ret, beamerOutput = showHight.show_Hights(calculationOutput)
            if not ret:
                continue  # Wenn die Bildkonvertierung fehlschlägt, überspringe diesen Frame
        
            yield beamerOutput
    finally:
        # Ressourcen freigeben
        pipeline.stop()
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
            #ret, beamerOutput = showColorAndDepth.show_Color_And_Depth(cv2.convertScaleAbs(color_and_depth_image["depth"], alpha=0.03), color_and_depth_image["color"])
            ret, beamerOutput = showColorAndDepth.show_Color_And_Depth(np.uint8(color_and_depth_image["depth"]), color_and_depth_image["color"])
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
    VideoTheme(2, "2D Lautstärke TODO", volume_2D_video),
    VideoTheme(3, "RGB", color_video),
    VideoTheme(4, "Höhe", hights_video),
    VideoTheme(5, "Intel Real Sense", intel_video)

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
