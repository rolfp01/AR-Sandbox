import cv2
import numpy as np

# ===============================================
# Hilfsfunktionen für die Gebäudeerkennung
# ===============================================

def compute_height_from_depth(depth_image, depth_scale):
    """
    Berechnet die Höhe aus dem Tiefenbild unter Verwendung des depth_scale.
    Gibt gefilterte Höhe und gültige Pixelmaske zurück.
    """
    height_map = depth_image.astype(np.float32) * depth_scale
    valid = (height_map > 0)
    height_map_filtered = cv2.medianBlur(height_map, 5)
    return height_map_filtered, valid

def estimate_relative_height(height_map_filtered, valid):
    """
    Schätzt die relative Höhe basierend auf Höhe und gültigen Pixeln.
    """
    min_h = np.min(height_map_filtered[valid])
    max_h = np.max(height_map_filtered[valid])
    relative_height = np.zeros_like(height_map_filtered)
    if max_h == min_h:
        relative_height[valid] = 0
    else:
        relative_height[valid] = (height_map_filtered[valid] - min_h) / (max_h - min_h)

    return relative_height

def generate_building_candidates(relative_height, height_map_filtered, valid):
    """
    Erzeugt eine binäre Maske der Gebäudekandidaten basierend auf Schwellenwerten.
    """
    building_candidate = np.zeros_like(relative_height, dtype=np.uint8)
    
    # Höhen-Gradient berechnen
    grad_x = cv2.Sobel(height_map_filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map_filtered, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Schwelle für Kanten (Legosteinkanten)
    edge_mask = gradient_magnitude > 0.01
    
    # Beispiel: Gebäude, wenn relative Höhe > 0.3 und Höhe > 2 Meter
    building_candidate[((relative_height > 0.01) & (height_map_filtered > 0.1)) & edge_mask] = 255
    return building_candidate

def refine_building_mask(building_candidate):
    """
    Verbessert die Gebäudemaske durch morphologische Operationen.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    building_candidate = cv2.morphologyEx(building_candidate, cv2.MORPH_CLOSE, kernel)
    building_candidate = cv2.morphologyEx(building_candidate, cv2.MORPH_OPEN, kernel)
    return building_candidate

def filter_building_contours(building_candidate):
    """
    Filtert Gebäudekonturen nach Fläche und Form.
    """
    contours, _ = cv2.findContours(building_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    building_mask = np.zeros_like(building_candidate)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # Minimalfläche für Gebäude
            cv2.drawContours(building_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return building_mask

def smooth_building_mask(building_mask):
    """
    Glättet die Gebäudemaske, um Kanten zu mildern.
    """
    building_mask = cv2.GaussianBlur(building_mask, (3,3), 0)
    _, building_mask = cv2.threshold(building_mask, 127, 255, cv2.THRESH_BINARY)
    return building_mask

# ===============================================
# Hilfsfunktionen für die Straßenerkennung
# ===============================================

def extract_road_candidates(hsv):
    """
    Extrahiert potenzielle Straßenbereiche basierend auf Farbsegmentierung im HSV-Bereich.
    """
    # Beispiel für helle Grau- bis Weißtöne (Straßenfarbe)
    lower_road = np.array([90,30,30])
    upper_road = np.array([130,150,150])
    road_candidate_mask = cv2.inRange(hsv, lower_road, upper_road)
    return road_candidate_mask

def extract_sand_areas(hsv):
    """
    Extrahiert Sandbereiche aus dem HSV-Bild.
    """
    # Beispiel für sandige Farbtöne (gelblich)
    lower_sand = np.array([0,0,130])
    upper_sand = np.array([30,20,180])
    sand_mask = cv2.inRange(hsv, lower_sand, upper_sand)
    return sand_mask

def subtract_sand_from_road(road_mask, sand_mask, hsv):
    """
    Entfernt Sandbereiche aus der Straßenmaske.
    """
    # Optionale Verfeinerung: Schließe sandige Stellen aus
    road_mask[sand_mask > 0] = 0
    return road_mask

def postprocess_road_mask(road_mask):
    """
    Morphologische Nachbearbeitung der Straßenmaske.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    return road_mask

def filter_road_contours(road_mask):
    """
    Filtert Straßenkonturen basierend auf Fläche und Form.
    """
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(road_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return filtered_mask

def close_road_segments(road_mask):
    """
    Schließt kleine Lücken in Straßensegmenten.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    return road_mask

# ===============================================
# Hilfsfunktion für Schattenkorrektur
# ===============================================

def correct_shadow_effects(building_mask, road_mask):
    """
    Korrigiert Schatteneffekte: Entfernt Straßenbereiche, die unter Gebäudeschatten liegen.
    """
    shadow_region = cv2.dilate(building_mask, np.ones((15,15), np.uint8))
    road_mask[shadow_region > 0] = 0
    return road_mask

# ===============================================
# Hilfsfunktion für Park-Erkennung
# ===============================================

def detect_parks(color_image):
    """
    Erkennung von Parks (grüne Flächen) basierend auf Farbsegmentierung.
    """
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 60, 20])
    upper_green = np.array([90, 255, 255])
    park_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    park_mask = cv2.morphologyEx(park_mask, cv2.MORPH_CLOSE, kernel)
    park_mask = cv2.morphologyEx(park_mask, cv2.MORPH_OPEN, kernel)

    return park_mask

# ===============================================
# Hilfsfunktion zur Konfliktauflösung der Masken
# ===============================================

def resolve_mask_conflicts(building_mask, road_mask, park_mask):
    """
    Löst Konflikte zwischen Masken auf, z.B. überschneidende Bereiche.
    """
    # Gebäude haben höchste Priorität
    road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(building_mask))
    park_mask = cv2.bitwise_and(park_mask, cv2.bitwise_not(building_mask))
    # Parks haben Priorität über Straßen
    road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(park_mask))
    return road_mask, park_mask

# ===============================================
# Hilfsfunktion für zeitliche Filterung
# ===============================================

# Dummy Implementation – bitte anpassen, falls ein Objekt temporal gefiltert werden soll
def apply_temporal_filter(building_mask, road_mask, park_mask):
    """
    Filtert Masken zeitlich, um Rauschen zu reduzieren.
    """
    # Beispiel: keine zeitliche Filterung in dieser Version
    return building_mask, road_mask, park_mask

# ===============================================
# Hauptfunktion: detect_Buildings
# ===============================================

def detect_Buildings(depth_image, color_image, depth_scale, baseline_distance):
    """
    Objekterkennung: Erkennung von Gebäuden, Straßen und Parks 
    aus Tiefen- und Farb-Bildern.

    Parameter:
    - depth_image: Tiefenbild (numpy array)
    - color_image: Farb-Bild (BGR, numpy array)
    - depth_scale: Skalierungsfaktor für Tiefenwerte
    - baseline_distance: Abstand der Kameras (derzeit nicht genutzt)
    - temporal_filter: optionale zeitliche Filter-Informationen (default None)

    Rückgabe:
    - building_mask: Binärmaske für erkannte Gebäude
    - road_mask: Binärmaske für erkannte Straßen
    - park_mask: Binärmaske für erkannte Parks
    """

    # Leere Masken vorbereiten
    building_mask = np.zeros_like(depth_image, dtype=np.uint8)
    road_mask = np.zeros_like(depth_image, dtype=np.uint8)
    park_mask = np.zeros_like(depth_image, dtype=np.uint8)

    # ========================================================================
    # GEBÄUDE-ERKENNUNG
    # ========================================================================

    # 1. Höhe aus Tiefenbild berechnen
    height_map_filtered, valid = compute_height_from_depth(depth_image, depth_scale)
    if not np.any(valid):
        return building_mask, road_mask, park_mask  # Kein gültiges Tiefenbild

    # 2. Relative Höhe ermitteln
    relative_height = estimate_relative_height(height_map_filtered, valid)

    # 3. Gebäude-Kandidaten mit Schwellenwerten erzeugen
    building_candidate = generate_building_candidates(relative_height, height_map_filtered, valid)

    # 4. Morphologische Filterung der Gebäudekandidaten
    building_candidate = refine_building_mask(building_candidate)

    # 5. Kontur-Analyse für endgültige Gebäudemasken
    building_mask = filter_building_contours(building_candidate)

    # 6. Glätten der finalen Gebäudemaske
    building_mask = smooth_building_mask(building_mask)

    # ========================================================================
    # STRASSEN-ERKENNUNG
    # ========================================================================

    # Farbkonvertierung
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # 1. Farbbasierte Masken
    road_candidate_mask = extract_road_candidates(hsv)

    # 2. Sandbereiche herausfiltern
    sand_mask = extract_sand_areas(hsv)
    road_candidate_mask = subtract_sand_from_road(road_candidate_mask, sand_mask, hsv)

    # 3. Morphologische Nachbearbeitung
    road_mask_clean = postprocess_road_mask(road_candidate_mask)

    # 4. Konturfilterung für Straßen
    road_mask = filter_road_contours(road_mask_clean)

    # 5. Finale Verbindung der Straßen
    road_mask = close_road_segments(road_mask)

    # ========================================================================
    # SCHATTEN-KORREKTUR
    # ========================================================================

    road_mask = correct_shadow_effects(building_mask, road_mask)

    # ========================================================================
    # PARK-ERKENNUNG
    # ========================================================================

    park_mask = detect_parks(color_image)

    # ========================================================================
    # KONFLIKT-AUFLÖSUNG DER MASKEN
    # ========================================================================

    road_mask, park_mask = resolve_mask_conflicts(building_mask, road_mask, park_mask)

    # ========================================================================
    # ZEITLICHE FILTERUNG
    # ========================================================================

    #building_mask, road_mask, park_mask = apply_temporal_filter(
    #    building_mask, road_mask, park_mask
    #)

    return building_mask, road_mask, park_mask
