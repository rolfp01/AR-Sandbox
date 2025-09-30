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
    depth_array = depth_image.astype(np.float32) * depth_scale
    
    # Höhenkarte berechnen
    valid = depth_array > 0
    if not np.any(valid):
        return building_mask, road_mask, park_mask
        
    max_depth = np.max(depth_array[valid])
    height_map = max_depth - depth_array
    height_map[~valid] = 0
    
    # Hintergrund-Schätzung mit mittlerem Kernel
    background = cv2.GaussianBlur(height_map, (21, 21), 0)
    relative_height = height_map - background
    
    # NIEDRIGERE Schwellwerte - wichtig für LEGO-Erkennung!
    height_threshold = 0.006  # 6mm relative Höhe (noch niedriger)
    abs_height_threshold = 0.015  # 1.5cm absolute Höhe (noch niedriger)
    
    # Binäre Maske für erhöhte Objekte
    building_candidate = ((relative_height > height_threshold) & 
                          (height_map > abs_height_threshold)).astype(np.uint8) * 255
    
    # WICHTIG: Erst CLOSE, dann OPEN - verhindert Fragmentierung!
    kernel_close = np.ones((9, 9), np.uint8)  # Größerer Kernel zum Zusammenfügen
    kernel_open = np.ones((3, 3), np.uint8)   # Kleiner Kernel für Rauschen
    
    # Erst Lücken schließen (CLOSE)
    building_clean = cv2.morphologyEx(building_candidate, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # Dann kleine Störungen entfernen (OPEN)
    building_clean = cv2.morphologyEx(building_clean, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Konturfilterung
    contours, _ = cv2.findContours(building_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Sehr niedrige Mindestgröße
        if area < 30:  # Noch niedriger!
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        # Kompaktheit: 1.0 = perfekter Kreis, niedrigere Werte = eckiger
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        
        # Sehr tolerante Form-Kriterien
        if compactness < 0.85:  # Fast alles akzeptieren
            cv2.drawContours(building_mask, [cnt], -1, 255, -1)
        elif area > 200:  # Mittelgroße Objekte auch wenn runder
            cv2.drawContours(building_mask, [cnt], -1, 255, -1)
    
    # ZUSÄTZLICH: Nochmal CLOSE auf dem Endergebnis für zusammenhängende Gebäude
    building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)


    # *** Straßen-Erkennung auf Basis von Farbe/Form ***
    # 1. Farbbasierte Grau-Erkennung im HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # Maske für dunkle Flächen
    dark_mask = cv2.inRange(v_channel, 0, 120)

    # Sättigung
    s_channel = hsv[:, :, 1]
    mask_saturation = cv2.inRange(s_channel, 10, 100)

    # Kombiniere Masken: dunkel + moderate Sättigung (kein Schatten)
    road_candidate_mask = cv2.bitwise_and(dark_mask, mask_saturation)

    # Morphologische Filterung
    kernel = np.ones((3, 3), np.uint8)
    road_mask_clean = cv2.morphologyEx(road_candidate_mask, cv2.MORPH_OPEN, kernel)
    road_mask_clean = cv2.morphologyEx(road_mask_clean, cv2.MORPH_CLOSE, kernel)

    # Konturenerkennung wie gehabt
    road_mask = np.zeros_like(road_mask_clean)
    contours, _ = cv2.findContours(road_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w / h, h / w)

        if aspect_ratio > 1.0:
            cv2.drawContours(road_mask, [cnt], -1, 255, -1)

    # *** Park-Erkennung auf Basis von Farbe ***
    # Annahme: Parks sind durch grüne Flächen repräsentiert (z.B. Grünfläche).
    # Segmentiere Grün-Töne im HSV-Farbraum.
    lower_green = np.array([30, 60, 20], dtype=np.uint8)   # untere Grenze für Grün (Hue ca. 35-85)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)  # Sättigung und Value recht hoch
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    green_lab_mask = cv2.inRange(a, 80, 140)  # experimentell anpassen

    # Kombiniere HSV und LAB
    park_mask = cv2.bitwise_and(mask_green, green_lab_mask)

    return building_mask, road_mask, park_mask