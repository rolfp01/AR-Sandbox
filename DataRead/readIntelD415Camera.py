import numpy as np
import pyrealsense2 as rs
import cv2
from templates.base_camera_manager import BaseCameraManager

class IntelD415CameraManager(BaseCameraManager):
    """Manager für Intel RealSense D415 Kamera"""
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.config = None
    
    def start(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_scale = 0.001  # RealSense üblich

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
        self.config.enable_stream(
            rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Pipeline starten
        self.pipeline.start(self.config)
        print("Intel RealSense D415 erfolgreich initialisiert")
    
    def read_frame(self):
        return read_Intel_Camera_optimized(self.pipeline)
    
    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as e:
                print(f"Fehler beim Stoppen der Pipeline: {e}")
        cv2.destroyAllWindows()
        print("Intel RealSense D415 gestoppt")

def read_Intel_Camera(pipeline):
    # Frames abrufen und alignieren
    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not depth_frame or not color_frame:
        return None
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Linken Rand korrigieren (falls nötig)
    if depth_image.shape[1] > 60:
        depth_image[:, :60] = 0
        color_image[:, :60] = 0
    
    return {
        "color": color_image, 
        "depth": depth_image
    }


def read_Intel_Camera_optimized(pipeline):
    """
    OPTIMIERTE Kamera-Auslesung - OHNE Warmup
    Fokus auf maximale Bildqualität und Stabilität
    """
    
    # Frames abrufen und alignieren
    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not depth_frame or not color_frame:
        return None
    
    # ========================================================================
    # OPTIMIERTE FILTER-PIPELINE
    # ========================================================================
    
    # 1. DECIMATION - reduziert Auflösung für bessere Performance (optional)
    # Kommentiere aus, wenn du volle Auflösung willst
    # decimation = rs.decimation_filter()
    # decimation.set_option(rs.option.filter_magnitude, 2)
    # depth_frame = decimation.process(depth_frame)
    
    # 2. THRESHOLD - Bereich begrenzen (ZUERST!)
    threshold_filter = rs.threshold_filter()
    threshold_filter.set_option(rs.option.min_distance, 0.6)  # 50cm
    threshold_filter.set_option(rs.option.max_distance, 0.9)   # 90cm
    depth_frame = threshold_filter.process(depth_frame)
    
    # 3. DISPARITY TRANSFORM - für bessere Filter-Performance
    depth_to_disparity = rs.disparity_transform(True)
    depth_frame = depth_to_disparity.process(depth_frame)
    
    # 4. SPATIAL FILTER - reduziert räumliches Rauschen
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)      # Stärker filtern
    spatial.set_option(rs.option.filter_smooth_alpha, 0.6) # Höhere Glättung
    spatial.set_option(rs.option.filter_smooth_delta, 25)  # Delta erhöht
    spatial.set_option(rs.option.holes_fill, 3)            # Loch-Füllung
    depth_frame = spatial.process(depth_frame)
    
    # 5. TEMPORAL FILTER - reduziert zeitliches Rauschen (Flackern!)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.5)  # Mittelstark
    temporal.set_option(rs.option.filter_smooth_delta, 25)
    depth_frame = temporal.process(depth_frame)
    
    # 6. ZURÜCK ZU DEPTH
    disparity_to_depth = rs.disparity_transform(False)
    depth_frame = disparity_to_depth.process(depth_frame)
    
    # 7. HOLE FILLING - füllt verbleibende Löcher
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)  # Farthest-from-around
    depth_frame = hole_filling.process(depth_frame)
    
    # ========================================================================
    # NUMPY ARRAYS ERSTELLEN
    # ========================================================================
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Linken Rand korrigieren (falls nötig)
    if depth_image.shape[1] > 60:
        depth_image[:, :60] = 0
    
    # ZUSÄTZLICHE POST-PROCESSING (optional - für noch bessere Qualität)
    
    # Median-Filter auf Depth (entfernt Salz-Pfeffer-Rauschen)
    depth_image = cv2.medianBlur(depth_image.astype(np.uint16), 5)
    
    # Bilateral Filter auf Color (behält Kanten, glättet Flächen)
    color_image = cv2.bilateralFilter(
        color_image, d=5, sigmaColor=50, sigmaSpace=50)
    
    return {
        "color": color_image, 
        "depth": depth_image
    }
