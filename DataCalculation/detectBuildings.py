import numpy as np
import cv2

def detect_Buildings(depth_image, color_image, depth_scale, baseline_distance):
    """
    Erkenne Objekte (Gebäude, Straßen, Parks) mittels klassischer Bildverarbeitung.
    Returns:
        building_mask, road_mask, park_mask - binäre Masken (NumPy-Arrays vom Typ uint8),
        wobei Pixelwert 255 anzeigt, dass dort das entsprechende Objekt erkannt wurde.
    """
    # Initialisiere leere Masken
    building_mask = np.zeros_like(depth_image, dtype=np.uint8)
    road_mask = np.zeros_like(depth_image, dtype=np.uint8)
    park_mask = np.zeros_like(depth_image, dtype=np.uint8)

    # *** Gebäude-Erkennung auf Basis der Höhe (Tiefenbild) ***
    # Wir nehmen an, "Gebäude" sind Bereiche, die deutlich über der umgebenden Sandhöhe liegen.
    # Ansatz: Schwellwert-Filter auf das Tiefenbild (niedriger Abstand => hohe Struktur).
    # Berechne eine relative Höhenkarte (inverse Tiefe). 
    # Wenn baseline_distance bekannt, könnte man hier die absolute Höhe berechnen.
    depth_array = depth_image.astype(np.float32) * depth_scale  # Tiefenwerte in Meter
    if baseline_distance is not None:
        # Höhe = Abstand Kamera zum Boden - aktueller Abstand (Meter)
        height_map = (baseline_distance - depth_array)
    else:
        # Ohne bekannten Boden nehmen wir relative Höhe: invertiere aktuelle Tiefenwerte relativ zum Bild
        # (Kleinerer Tiefenwert => größere Höhe über minimaler Tiefe im Bild)
        valid = depth_array > 0
        if np.any(valid):
            min_depth = depth_array[valid].min()
            max_depth = depth_array[valid].max()
        else:
            min_depth, max_depth = 0, 0
        height_map = (max_depth - depth_array)  # invers zur Tiefe

    # Normiere Höhe auf 0-255 (für Schwellwertberechnung)
    height_norm = cv2.normalize(src=height_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # Schwellwert setzen (z.B. oberste 20% der Höhe als Gebäude markieren)
    _, building_mask = cv2.threshold(height_norm, 204, 255, cv2.THRESH_BINARY)
    # Morphologische Öffnung anwenden, um Rauschen zu entfernen (entfernt kleine Pixel) [oai_citation_attribution:5‡docs.opencv.org](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html#:~:text=3)
    kernel = np.ones((3, 3), np.uint8)
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN, kernel)

    # * Straßen-Erkennung auf Basis von Farbe/Form *
    # Annahme: Straßen haben charakteristische graue/dunkle Farbe und liegen flach auf Bodenhöhe.
    # Wir nutzen Farbbild in HSV-Farbraum, um graue Bereiche zu segmentieren.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # Definiere HSV-Schwellenwerte für "grau"/"asphaltiert":
    # - Hue: egal (0-179), Sättigung niedrig (wenig Farbe), Value mittel bis hoch (nicht zu dunkel oder zu hell).
    lower_gray = np.array([0, 0, 50], dtype=np.uint8)
    upper_gray = np.array([140, 140, 140], dtype=np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)  # liefert binäre Maske der grauen Pixel [oai_citation_attribution:6‡realpython.com](https://realpython.com/python-opencv-color-spaces/#:~:text=Once%20you%20get%20a%20decent,within%20the%20range%2C%20and%20zero)
    # Optional: Man könnte zusätzlich prüfen, ob diese Pixel nahe der Grundhöhe sind (per Tiefendaten),
    # um z.B. Schatten oder andere graue Objekte auszuschließen. Hier wird vereinfachend nur die Farbe genutzt.
    road_mask = mask_gray.copy()
    # Morphologische Operationen, um die Masken zu säubern (Rauschen entfernen, Lücken schließen):
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)   # kleine Störpixel entfernen [oai_citation_attribution:7‡docs.opencv.org](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html#:~:text=3)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)  # Lücken in Straßensegmenten schließen [oai_citation_attribution:8‡docs.opencv.org](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html#:~:text=4)

    # * Park-Erkennung auf Basis von Farbe *
    # Annahme: Parks sind durch grüne Flächen repräsentiert (z.B. Grünfläche).
    # Segmentiere Grün-Töne im HSV-Farbraum.
    lower_green = np.array([35, 60, 20], dtype=np.uint8)   # untere Grenze für Grün (Hue ca. 35-85)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)  # Sättigung und Value recht hoch
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    park_mask = mask_green.copy()
    # Morphologische Glättung (Öffnen/Schließen) für Parkmaske
    park_mask = cv2.morphologyEx(park_mask, cv2.MORPH_OPEN, kernel)
    park_mask = cv2.morphologyEx(park_mask, cv2.MORPH_CLOSE, kernel)

    return building_mask, road_mask, park_mask