from flask import Flask, redirect, render_template, Response, session, url_for, jsonify

from DataCalculation import calculate2DVolume, calculateHight, calculateRGB, detectBuildings, grayPicture
from DataRead import readAsusXtionCamera, readLaptopCamera, readIntelD415Camera
from DataShow import show2DVolume, showGrayPicture, showHight, showRGB, showObjects, showColorAndDepth
import numpy as np
import cv2
#from openni import openni2
import pyrealsense2 as rs
from primesense import openni2
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# am Programmstart (z.B. ganz oben)
OPENNI2_INITIALIZED = False

def init_openni2():
    global OPENNI2_INITIALIZED
    if not OPENNI2_INITIALIZED:
        openni_path = "C:\\Program Files\\OpenNI2\\Redist"
        openni2.initialize(openni_path)
        OPENNI2_INITIALIZED = True

def unload_openni2():
    global OPENNI2_INITIALIZED
    if OPENNI2_INITIALIZED:
        openni2.unload()
        OPENNI2_INITIALIZED = False

# ============================================================================
# Kamera-Konfiguration - HIER ÄNDERN!
# ============================================================================
# Wähle die Kamera für ALLE Themen:
# Optionen: 'laptop', 'asus_xtion', 'intel_d415', 'kinect
ACTIVE_CAMERA = 'asus_xtion'


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
    
class KinectCameraManager(BaseCameraManager):
    """Manager für Microsoft Kinect Kamera (Kinect v2) geht nur mit Python 3.8"""
    
    def __init__(self):
        super().__init__()
        self.kinect = None
    
    def start(self):
        from pykinect2 import PyKinectRuntime, PyKinectV2
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
        print("Microsoft Kinect erfolgreich initialisiert")
    
    def read_frame(self):
        if self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():
            color_frame = self.kinect.get_last_color_frame()
            depth_frame = self.kinect.get_last_depth_frame()
            
            # Color Frame in ein numpy Array umwandeln (BGRA 1920x1080)
            color_image = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
            # In BGR konvertieren für OpenCV (falls gewünscht)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            
            # Depth Frame (512x424) in numpy Array
            depth_image = depth_frame.reshape((424, 512)).astype(np.uint16)
            
            return {
                "color": color_image,
                "depth": depth_image
            }
        return None
    
    def stop(self):
        if self.kinect:
            self.kinect.close()
        cv2.destroyAllWindows()
        print("Microsoft Kinect gestoppt")


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
    """Manager für Asus XtionPRO Live Tiefenkamera über OpenNI2 (primesense)"""
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.color_stream = None
        self.depth_stream = None
        self.openni2_initialized = False

    def start(self):
        init_openni2()
        self.device = openni2.Device.open_any()
        if not self.device:
            raise RuntimeError("OpenNI2 Gerät konnte nicht geöffnet werden")

        self.color_stream = self.device.create_color_stream()
        if not self.color_stream:
            raise RuntimeError("Farb-Stream konnte nicht erstellt werden")
        self.color_stream.start()

        self.depth_stream = self.device.create_depth_stream()
        if not self.depth_stream:
            raise RuntimeError("Tiefen-Stream konnte nicht erstellt werden")
        self.depth_stream.start()

        print("Asus XtionPRO Live erfolgreich initialisiert (OpenNI2/primesense)")

    def read_frame(self):
        if not self.color_stream or not self.depth_stream:
            print("[WARNUNG] Kamera-Streams nicht aktiv")
            return None

        try:
            # Farbbild lesen
            c_frame = self.color_stream.read_frame()
            width, height = c_frame.width, c_frame.height
            c_data = c_frame.get_buffer_as_uint8()

            if len(c_data) == width * height * 3:
                color = np.frombuffer(c_data, dtype=np.uint8).reshape((height, width, 3))
                # RGB zu BGR konvertieren (für OpenCV)
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            elif len(c_data) == width * height:
                # 1 Kanal (Graustufen)
                color = np.frombuffer(c_data, dtype=np.uint8).reshape((height, width))
                color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
            else:
                print(f"[WARNUNG] Unerwartete Farbbildgröße: {len(c_data)} Bytes")
                return None

            # Tiefenbild lesen
            d_frame = self.depth_stream.read_frame()
            depth = np.frombuffer(d_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((d_frame.height, d_frame.width))

            return {
                "color": color,
                "depth": depth
            }

        except Exception as e:
            print(f"[FEHLER] Fehler beim Lesen des Kamera-Frames: {e}")
            return None

    def stop(self):
        if self.color_stream:
            self.color_stream.stop()
            self.color_stream = None
        if self.depth_stream:
            self.depth_stream.stop()
            self.depth_stream = None
        # Nicht hier openni2.unload() aufrufen!
        cv2.destroyAllWindows()
        print("Asus XtionPRO Live gestoppt")

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
    elif camera_type == "kinect":
        return KinectCameraManager()
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


def process_double_video(camera, frame_data):
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
    VideoTheme(5, "Doppel Bild", process_double_video)
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