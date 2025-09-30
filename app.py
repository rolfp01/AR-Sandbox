from flask import Flask, redirect, render_template, Response, session, url_for, jsonify

from DataCalculation import calculate2DVolume, calculateHight, calculateRGB, detectBuildings, grayPicture
from DataRead import readAsusXtionCamera, readLaptopCamera, readIntelD415Camera
from DataShow import show2DVolume, showGrayPicture, showHight, showRGB, showObjects, showColorAndDepth
import numpy as np
import cv2
from openni import openni2
import pyrealsense2 as rs

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ============================================================================
# Kamera-Konfiguration - HIER ÄNDERN!
# ============================================================================
# Wähle die Kamera für ALLE Themen:
# Optionen: 'laptop', 'asus_xtion', 'intel_d415'
ACTIVE_CAMERA = 'intel_d415'


# ============================================================================
# Kamera-Manager-Klassen
# ============================================================================

class BaseCameraManager:
    """Basis-Klasse für alle Kamera-Manager"""
    
    def __init__(self):
        self.depth_scale = 1
        self.baseline_distance = None
    
    def start(self):
        """Startet die Kamera - muss von Unterklassen implementiert werden"""
        raise NotImplementedError
    
    def read_frame(self):
        """Liest ein Frame - muss von Unterklassen implementiert werden"""
        raise NotImplementedError
    
    def stop(self):
        """Stoppt die Kamera - muss von Unterklassen implementiert werden"""
        raise NotImplementedError


class LaptopCameraManager(BaseCameraManager):
    """Manager für Laptop-Kamera (Webcam)"""
    
    def __init__(self):
        super().__init__()
        self.camera = None
    
    def start(self):
        self.camera = cv2.VideoCapture(0)
        print("Laptop-Kamera erfolgreich initialisiert")
    
    def read_frame(self):
        success, frame = self.camera.read()
        if success:
            color_image = readLaptopCamera.read_Laptop_Camera(frame)
            # Rückgabe im gleichen Format wie andere Kameras
            return {
                "color": color_image,
                "depth": None  # Laptop-Kamera hat keine Tiefendaten
            }
        return None
    
    def stop(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Laptop-Kamera gestoppt")


class AsusXtionCameraManager(BaseCameraManager):
    """Manager für Asus Xtion Kamera"""
    
    def __init__(self):
        super().__init__()
        self.dev = None
        self.color_stream = None
        self.depth_stream = None
    
    def start(self):
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.start()
        print("Asus Xtion erfolgreich initialisiert")
    
    def read_frame(self):
        color_image = readAsusXtionCamera.read_Depth_Camera_only_color(self.color_stream)
        return {
            "color": color_image,
            "depth": None  # Kann erweitert werden, wenn Tiefendaten benötigt werden
        }
    
    def stop(self):
        if self.color_stream:
            self.color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        print("Asus Xtion gestoppt")


class IntelD415CameraManager(BaseCameraManager):
    """Manager für Intel RealSense D415 Kamera"""
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.config = None
    
    def start(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Geräte-Konfiguration
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        # RGB-Sensor prüfen
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        
        if not found_rgb:
            raise RuntimeError("Intel D415: RGB-Kamera nicht gefunden")
        
        # Streams aktivieren
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Pipeline starten
        self.pipeline.start(self.config)
        print("Intel RealSense D415 erfolgreich initialisiert")
    
    def read_frame(self):
        return readIntelD415Camera.read_Depth_Camera(self.pipeline)
    
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Intel RealSense D415 gestoppt")


# ============================================================================
# Kamera-Factory
# ============================================================================

def create_camera_manager(camera_type):
    """Erstellt den passenden Kamera-Manager basierend auf dem Typ"""
    if camera_type == "laptop":
        return LaptopCameraManager()
    elif camera_type == "asus_xtion":
        return AsusXtionCameraManager()
    elif camera_type == "intel_d415":
        return IntelD415CameraManager()
    else:
        raise ValueError(f"Unbekannter Kamera-Typ: {camera_type}")


# ============================================================================
# Video-Verarbeitungsfunktionen
# ============================================================================

def process_video_stream(processing_function):
    """
    Generische Video-Stream-Funktion
    
    Args:
        processing_function: Funktion zur Frame-Verarbeitung
    """
    camera = create_camera_manager(ACTIVE_CAMERA)
    
    try:
        camera.start()
        
        while True:
            try:
                # Frame von Kamera lesen
                frame_data = camera.read_frame()
                
                if frame_data is None:
                    continue
                
                # Verarbeitung durchführen
                beamer_output = processing_function(camera, frame_data)
                
                # Ausgabe generieren
                if beamer_output is not None:
                    yield beamer_output
                    
            except Exception as e:
                print(f"Fehler bei Frame-Verarbeitung: {e}")
                continue
                
    finally:
        camera.stop()


# ============================================================================
# Spezifische Verarbeitungsfunktionen
# ============================================================================

def process_gray_video(camera, frame_data):
    """Verarbeitet Graustufen-Video"""
    if frame_data["color"] is None:
        return None
    calculation_output = grayPicture.picture_In_Gray(frame_data["color"])
    return showGrayPicture.show_Gray_Picture(calculation_output)


def process_color_video(camera, frame_data):
    """Verarbeitet RGB-Video"""
    if frame_data["color"] is None:
        return None
    calculation_output = calculateRGB.calculate_Colors(frame_data["color"])
    ret, beamer_output = showRGB.show_Colors(calculation_output)
    return beamer_output if ret else None


def process_objects_video(camera, frame_data):
    """Verarbeitet Objekt-Erkennung"""
    if frame_data["depth"] is None or frame_data["color"] is None:
        return None
    
    building_mask, road_mask, park_mask = detectBuildings.detect_Buildings(
        frame_data["depth"], 
        frame_data["color"], 
        camera.depth_scale, 
        camera.baseline_distance
    )
    ret, beamer_output = showObjects.show_Objects(building_mask, road_mask, park_mask)
    return beamer_output if ret else None


def process_volume_2d_video(camera, frame_data):
    """Verarbeitet 2D-Volumen"""
    if frame_data["depth"] is None or frame_data["color"] is None:
        return None
    
    building_mask, road_mask, park_mask = detectBuildings.detect_Buildings(
        frame_data["depth"], 
        frame_data["color"], 
        camera.depth_scale, 
        camera.baseline_distance
    )
    calculation_output = calculate2DVolume.calculate_2D_Volume(
        frame_data['depth'], 
        building_mask, 
        road_mask, 
        park_mask
    )
    ret, beamer_output = show2DVolume.show_2D_Volume(calculation_output)
    return beamer_output if ret else None


def process_heights_video(camera, frame_data):
    """Verarbeitet Höhen-Video"""
    if frame_data["depth"] is None:
        return None
    
    calculation_output = calculateHight.calculate_Hight(frame_data["depth"])
    ret, beamer_output = showHight.show_Hights(calculation_output)
    return beamer_output if ret else None


def process_intel_raw_video(camera, frame_data):
    """Verarbeitet rohes Intel RealSense Video"""
    if frame_data["depth"] is None or frame_data["color"] is None:
        return None
    
    ret, beamer_output = showColorAndDepth.show_Color_And_Depth(
        np.uint8(frame_data["depth"]), 
        frame_data["color"]
    )
    return beamer_output if ret else None


# ============================================================================
# Video-Themen Definition
# ============================================================================

class VideoTheme:
    """Repräsentiert ein Video-Verarbeitungs-Thema"""
    
    def __init__(self, index, name, process_func):
        self.index = index
        self.name = name
        self.process_func = process_func
    
    def get_stream(self):
        """Gibt den Video-Stream für dieses Thema zurück"""
        return process_video_stream(self.process_func)


# Liste aller verfügbaren Video-Themen
videoThemes = [
    VideoTheme(0, "Graustufen Video", process_gray_video),
    VideoTheme(1, "Objekte", process_objects_video),
    VideoTheme(2, "2D Volumen", process_volume_2d_video),
    VideoTheme(3, "RGB", process_color_video),
    VideoTheme(4, "Höhe", process_heights_video),
    VideoTheme(5, "Intel RealSense Raw", process_intel_raw_video)
]


# ============================================================================
# Flask-Routen
# ============================================================================

@app.route('/')
def index():
    """Startseite"""
    if 'activeVideoTheme' not in session:
        session['activeVideoTheme'] = 0
    
    current_theme = videoThemes[session['activeVideoTheme']].name
    
    return render_template('index.html', 
                         current_theme=current_theme,
                         active_camera=ACTIVE_CAMERA)


@app.route('/video_feed')
def video_feed():
    """Video-Stream-Endpunkt"""
    theme_index = session.get('activeVideoTheme', 0)
    return Response(
        videoThemes[theme_index].get_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/theme_Switch')
def theme_Switch():
    """Wechselt zum nächsten Video-Thema"""
    current = session.get('activeVideoTheme', 0)
    session['activeVideoTheme'] = (current + 1) % len(videoThemes)
    
    theme_name = videoThemes[session['activeVideoTheme']].name
    print(f"Thema gewechselt zu: {theme_name}")
    
    return redirect(url_for('index'))


@app.route('/set_theme/<int:theme_index>')
def set_theme(theme_index):
    """Setzt ein bestimmtes Video-Thema"""
    if 0 <= theme_index < len(videoThemes):
        session['activeVideoTheme'] = theme_index
        theme_name = videoThemes[theme_index].name
        print(f"Thema gesetzt zu: {theme_name}")
    return redirect(url_for('index'))


@app.route('/camera_info')
def camera_info():
    """Gibt Informationen über die aktive Kamera zurück"""
    return jsonify({
        'active_camera': ACTIVE_CAMERA,
        'available_cameras': ['laptop', 'asus_xtion', 'intel_d415'],
        'themes': [theme.name for theme in videoThemes]
    })


# ============================================================================
# Hauptprogramm
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Video-Streaming-Server gestartet")
    print("=" * 70)
    print(f"\nAktive Kamera: {ACTIVE_CAMERA}")
    print(f"\nVerfügbare Themen ({len(videoThemes)}):")
    for theme in videoThemes:
        print(f"  {theme.index}. {theme.name}")
    print("\nÄndern Sie ACTIVE_CAMERA oben in der Datei,")
    print("um eine andere Kamera zu verwenden!")
    print("=" * 70)
    
    app.run(debug=True)