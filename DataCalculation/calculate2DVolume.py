import numpy as np
import cv2

def calculate_2D_Volume(depth_image, building_mask, road_mask, park_mask):
    """
    Simuliere eine Lärmverteilung unter Berücksichtigung von Straßen, Parks und Gebäuden.
    Gebäude blockieren oder reduzieren die Ausbreitung von Lärm.
    Gibt eine 8-Bit Grauwert-Karte zurück (0 = kein Lärm, 255 = maximale Lautstärke).
    """
    # Konvertiere Masken zu Gleitkommazahlen (0.0 bis 1.0)
    road_src = road_mask.astype(np.float32) / 255.0
    park_src = park_mask.astype(np.float32) / 255.0
    building_src = building_mask.astype(np.float32) / 255.0

    # Definiere Lärmquellenintensitäten
    road_intensity = 1.0
    park_intensity = 0.6

    # Initialisiere Lärmquellekarte
    noise_sources = np.zeros_like(road_src, dtype=np.float32)
    noise_sources += road_intensity * road_src
    noise_sources += park_intensity * park_src

    # Wende mehrfache Weichzeichnungen an, aber verhindere, dass Lärm durch Gebäude "wandert"
    # Dafür nutzen wir eine Diffusion mit Dämpfung durch Gebäude
    noise = noise_sources.copy()
    for i in range(20):  # mehrere Iterationen zur Simulation von Ausbreitung
        blurred = cv2.GaussianBlur(noise, (91, 91), sigmaX=0)
        
        # Verhindere Übertragung durch Gebäude – dort wird nicht erhöht
        noise = np.where(building_src > 0.5, noise, blurred)

        # Optional: Original-Lärmquellen beibehalten (damit sie nicht "ausgewaschen" werden)
        noise = np.maximum(noise, noise_sources)

    # Normiere und konvertiere zu 8-Bit Bild
    noise_norm = cv2.normalize(noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    noise_map = noise_norm.astype(np.uint8)

    return noise_map
