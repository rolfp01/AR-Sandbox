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
    
    # Option 1: GaussianBlur (schnell, weiche Glättung)
    height_map_filtered = cv2.GaussianBlur(height_map, (7, 7), 0)
    
    # Option 2: Bilateral Filter (langsamer, erhält Kanten besser)
    # height_map_filtered = cv2.bilateralFilter(height_map, 9, 75, 75)
    
    # Option 3: Median-ähnlich mit scipy (wenn verfügbar, beste Ausreißer-Unterdrückung)
    # from scipy.ndimage import median_filter
    # height_map_filtered = median_filter(height_map, size=7)
    
    return height_map_filtered, valid

def estimate_relative_height(height_map_filtered, valid):
    """
    Schätzt die relative Höhe basierend auf Höhe und gültigen Pixeln.
    """
    relative_height = np.zeros_like(height_map_filtered)
    
    # Globale relative Höhe
    if np.any(valid):
        min_h = np.min(height_map_filtered[valid])
        max_h = np.max(height_map_filtered[valid])
        if max_h > min_h:
            relative_height[valid] = (height_map_filtered[valid] - min_h) / (max_h - min_h)
    
    return relative_height

def compute_local_height_difference(height_map_filtered, valid):
    """
    Berechnet lokale Höhenunterschiede zur Umgebung.
    Dies erkennt Bau-Steine auch auf Hügeln, da nur der Sprung gemessen wird.
    
    BEISPIEL bei hügeligem Untergrund:
    - Flacher Sand bei 0cm -> Bauklotz 5cm -> Differenz = 5cm ✓
    - Hügel bei 10cm -> Bauklotz auf Hügel 15cm -> Differenz = 5cm ✓
    - Sanfter Hügel 0-10cm -> Differenz ~0cm (kein Bauklotz) ✗
    
    Der lokale Durchschnitt (Blur) repräsentiert die "Hügel-Höhe".
    Die Differenz zeigt nur Objekte, die SPITZ aus der Umgebung herausragen.
    """
    # Lokaler Durchschnitt der Umgebung (größerer Kernel für sanfte Hügel)
    kernel_size = 21  # Muss ungerade sein, größer = erkennt sanftere Hügel
    local_mean = cv2.blur(height_map_filtered, (kernel_size, kernel_size))
    
    # Höhendifferenz: Wie viel höher ist jeder Punkt als seine Umgebung?
    height_difference = height_map_filtered - local_mean
    
    # Nur positive Unterschiede (Objekte die HÖHER sind als Umgebung)
    height_difference[height_difference < 0] = 0
    height_difference[~valid] = 0
    
    return height_difference

def generate_building_candidates(relative_height, height_map_filtered, valid, height_difference):
    """
    Erzeugt eine binäre Maske der Gebäudekandidaten basierend auf Schwellenwerten.
    OPTIMIERT für AR Sandbox mit hügeligem Untergrund.
    """
    building_candidate = np.zeros_like(relative_height, dtype=np.uint8)
    
    # STRATEGIE 1: Lokale Höhendifferenz (funktioniert bei Hügeln!)
    # Ein Bau-Stein ist 1-19cm höher als seine unmittelbare Umgebung
    local_height_mask = (height_difference >= 0.01) & (height_difference <= 0.20)
    
    # STRATEGIE 2: Absolute Höhe (nur als Backup für flachen Untergrund)
    # Funktioniert wenn Sand relativ flach ist
    absolute_height_mask = (height_map_filtered >= 0.01) & (height_map_filtered <= 0.20)
    
    # STRATEGIE 3: Relative Höhe (für sehr variable Szenen)
    relative_mask = relative_height > 0.15
    
    # Kombiniere: Hauptsächlich lokale Differenz, aber auch andere akzeptieren
    # ODER-Verknüpfung: Mindestens eine Strategie muss zutreffen
    building_candidate[(local_height_mask | (absolute_height_mask & relative_mask)) & valid] = 255
    
    # Entferne kleine Rauschpunkte
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    building_candidate = cv2.morphologyEx(building_candidate, cv2.MORPH_OPEN, kernel_noise)
    
    return building_candidate

def refine_building_mask(building_candidate):
    """
    Verbessert die Gebäudemaske durch morphologische Operationen.
    OPTIMIERT: Füllt Gebäude vollständig aus, nicht nur Ränder.
    """
    # 1. Schließe kleine Lücken innerhalb der Gebäude
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    building_candidate = cv2.morphologyEx(building_candidate, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Entferne kleine Rauschregionen außerhalb
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    building_candidate = cv2.morphologyEx(building_candidate, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Dilatation um sicherzustellen, dass Gebäude zusammenhängend sind
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    building_candidate = cv2.dilate(building_candidate, kernel_dilate, iterations=1)
    
    return building_candidate

def filter_building_contours(building_candidate):
    """
    Filtert Gebäudekonturen nach Fläche und Form.
    OPTIMIERT: Akzeptiert verschiedene Klötzchen-Gebäudegrößen.
    """
    contours, _ = cv2.findContours(building_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    building_mask = np.zeros_like(building_candidate)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Sehr breiter Bereich für verschiedene Klötzchen-Größen
        # Minimum: kleine 2x2 Gebäude-Steine (~30 Pixel)
        # Maximum: große Konstruktionen (~15000 Pixel)
        if 30 < area < 15000:
            # Fülle die Kontur vollständig
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
    Extrahiert potenzielle Straßenbereiche (dunkler Papierstreifen) im HSV-Bereich.
    OPTIMIERT für AR Sandbox mit dunkelgrauem Papier.
    """
    # Beispiel für helle Grau- bis Weißtöne (Straßenfarbe)
    lower_road = np.array([90,30,30])
    upper_road = np.array([130,150,150])
    road_candidate_mask = cv2.inRange(hsv, lower_road, upper_road)
    return road_candidate_mask

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
    OPTIMIERT für langen Papierstreifen in AR Sandbox.
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
    Erkennung von Parks (grüne Papierschnipsel) basierend auf Farbsegmentierung.
    OPTIMIERT für AR Sandbox mit grünem Papier.
    """
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # Grüne Farbbereiche (breiter für verschiedene Grüntöne)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    park_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphologie: Schließe Lücken zwischen Papierschnipseln
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    park_mask = cv2.morphologyEx(park_mask, cv2.MORPH_CLOSE, kernel)
    park_mask = cv2.morphologyEx(park_mask, cv2.MORPH_OPEN, kernel)
    
    # Entferne sehr kleine Schnipsel (Rauschen)
    contours, _ = cv2.findContours(park_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_park = np.zeros_like(park_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Mindestgröße für Papierschnipsel
            cv2.drawContours(filtered_park, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_park

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
# Debug-Hilfsfunktion (optional)
# ===============================================

def visualize_detection_debug(color_image, depth_image, building_mask, road_mask, park_mask, 
                               height_map_filtered=None, show=True):
    """
    Visualisiert die Erkennungsergebnisse zur Fehleranalyse.
    Zeigt Original, Tiefenkarte und alle drei Masken.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Zeile 1: Eingaben
    axes[0, 0].imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    if height_map_filtered is not None:
        im = axes[0, 1].imshow(height_map_filtered, cmap='jet')
        axes[0, 1].set_title('Höhenkarte (m)')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
    
    axes[0, 2].imshow(depth_image, cmap='gray')
    axes[0, 2].set_title('Tiefenbild')
    axes[0, 2].axis('off')
    
    # Zeile 2: Erkennungsergebnisse
    axes[1, 0].imshow(building_mask, cmap='Reds')
    axes[1, 0].set_title('Gebäude (Rot)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(road_mask, cmap='Greys')
    axes[1, 1].set_title('Straßen (Grau)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(park_mask, cmap='Greens')
    axes[1, 2].set_title('Parks (Grün)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

# ===============================================
# Hauptfunktion: detect_Buildings
# ===============================================

def detect_Buildings(depth_image, color_image, depth_scale, baseline_distance, debug=False):
    """
    Objekterkennung für AR Sandbox: Erkennung von Gebäuden, Straßen und Parks.
    
    OPTIMIERT FÜR:
    - Gebäude: Eckiger Bauklotz (Höhe 1-19 cm)
    - Straßen: Dunkler grauer Papierstreifen
    - Parks: Grüne Papierschnipsel
    
    FUNKTIONIERT MIT HÜGELIGEM UNTERGRUND:
    Verwendet lokale Höhendifferenzen statt absoluter Höhe.
    Ein 5cm Gebäudestein wird erkannt, egal ob auf flachem Sand (5cm absolut) 
    oder auf 10cm Hügel (15cm absolut, aber +5cm lokal).
    
    WICHTIGE PARAMETER ZUM ANPASSEN:
    - kernel_size in compute_local_height_difference(): 
      * Größer (z.B. 31) = ignoriert größere Hügel
      * Kleiner (z.B. 11) = empfindlicher, aber erkennt kleine Hügel als Gebäude
    - height_difference Schwelle (0.008-0.20m):
      * Muss Bauklötzchen-Höhe entsprechen, nicht absolute Höhe!

    Parameter:
    - depth_image: Tiefenbild (numpy array)
    - color_image: Farb-Bild (BGR, numpy array)
    - depth_scale: Skalierungsfaktor für Tiefenwerte
    - baseline_distance: Abstand der Kameras (derzeit nicht genutzt)
    - debug: Wenn True, gibt zusätzlich die Höhenkarte zurück (default: False)

    Rückgabe:
    - building_mask: Binärmaske für erkannte Gebäude
    - road_mask: Binärmaske für erkannte Straßen
    - park_mask: Binärmaske für erkannte Parks
    - (height_map_filtered): Nur wenn debug=True
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

    # 2. Relative Höhe ermitteln (global)
    relative_height = estimate_relative_height(height_map_filtered, valid)
    
    # 3. Lokale Höhendifferenz berechnen (wichtig für Hügel!)
    height_difference = compute_local_height_difference(height_map_filtered, valid)

    # 4. Gebäude-Kandidaten mit mehreren Strategien erzeugen
    building_candidate = generate_building_candidates(relative_height, height_map_filtered, 
                                                      valid, height_difference)

    # 5. Morphologische Filterung der Gebäudekandidaten
    building_candidate = refine_building_mask(building_candidate)

    # 6. Kontur-Analyse für endgültige Gebäudemasken
    building_mask = filter_building_contours(building_candidate)

    # 7. Glätten der finalen Gebäudemaske
    building_mask = smooth_building_mask(building_mask)

    # ========================================================================
    # STRASSEN-ERKENNUNG
    # ========================================================================

    # Farbkonvertierung
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # 1. Farbbasierte Masken
    road_candidate_mask = extract_road_candidates(hsv)

    # 2. Morphologische Nachbearbeitung
    road_mask_clean = postprocess_road_mask(road_candidate_mask)

    # 3. Konturfilterung für Straßen (mit Formanalyse)
    road_mask = filter_road_contours(road_mask_clean)

    # 4. Finale Verbindung der Straßen
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
    # DEBUG-AUSGABE (optional)
    # ========================================================================
    
    if debug:
        return building_mask, road_mask, park_mask, height_map_filtered
    
    return building_mask, road_mask, park_mask
