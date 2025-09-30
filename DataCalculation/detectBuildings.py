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
    depth_array = depth_image.astype(np.float32) * depth_scale  # Tiefenwerte in Meter

    # Invertiere Tiefe zur "Höhe"
    valid = depth_array > 0
    height_map = np.zeros_like(depth_array)
    if np.any(valid):
        max_depth = np.max(depth_array[valid])
        height_map = max_depth - depth_array  # je niedriger der Wert, desto höher das Objekt

    # Lokale Mittelwert-Glättung, um Umgebungshöhe zu schätzen
    blurred_height = cv2.blur(height_map, (15, 15))
    relative_height = height_map - blurred_height  # Positive Werte = über Umgebung

    # Binärmaske für "potenziell hohes Objekt"
    raw_building_mask = (relative_height > 0.02).astype(np.uint8) * 255  # ab 2 cm über Umgebung
    abs_height_mask = (height_map > 0.04).astype(np.uint8) * 255  # 4 cm
    raw_building_mask = cv2.bitwise_and(raw_building_mask, abs_height_mask)

    # Morphologische Glättung
    kernel = np.ones((5, 5), np.uint8)
    clean_building_mask = cv2.morphologyEx(raw_building_mask, cv2.MORPH_OPEN, kernel)
    clean_building_mask = cv2.morphologyEx(clean_building_mask, cv2.MORPH_CLOSE, kernel)

    # Konturbasierte Selektion: kompakte, große Objekte
    building_mask = np.zeros_like(clean_building_mask)
    contours, _ = cv2.findContours(clean_building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue  # zu klein

        # Kompaktheit: Kreisförmige Objekte (z. B. Hügel) sind meist weniger "eckig"
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        compactness = (4 * np.pi * area) / (perimeter ** 2)  # 1 = Kreis, < 1 = eckiger
        if compactness < 0.6:  # zu rund = eher Hügel
            cv2.drawContours(building_mask, [cnt], -1, 255, -1)

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