import numpy as np
import cv2

def calculate_2D_Volume(depth_image, building_mask, road_mask, park_mask):
# TODO

    """
    Simuliere eine Lärm-/Geräuschverteilung basierend auf Straßen- und Parkmasken.
    Gibt eine 8-Bit Grauwert-Karte zurück, die die relative Schalldruckintensität an jedem Ort darstellt.
    (0 = kein Lärm, 255 = maximale Lautstärke).
    """
    # Konvertiere Masken zu Gleitkomma für Berechnungen (0.0 oder 1.0)
    road_src = (road_mask.astype(np.float32) / 255.0)
    park_src = (park_mask.astype(np.float32) / 255.0)

    # Angenommen: Straßen erzeugen hohe Verkehrslärmintensität, Parks moderaten Personenlärm
    road_intensity = 1.0   # normierte Lärmquelle für Straße
    park_intensity = 0.6   # etwas leiserer Lärm von Park (Menschen, keine Maschinen)

    # Initiiere Lärmkarte mit Quellen: setze Straßenbereiche auf road_intensity, Parkbereiche auf park_intensity
    noise = np.zeros_like(road_src, dtype=np.float32)
    noise += road_intensity * road_src
    noise += park_intensity * park_src

    # Verteile den Lärm räumlich durch Weichzeichnung (simuliert Abklingen mit Entfernung)
    # Hier verwenden wir einen gaußschen Filter. Die Kernel-Größe bestimmt die Reichweite der Ausbreitung.
    noise_blur = cv2.GaussianBlur(noise, (51, 51), sigmaX=0, sigmaY=0)
    # (Ein großer 51x51-Kernel sorgt für weiträumiges Verstreichen; in einer realen Anwendung könnte
    # man die Ausbreitung z.B. proportional zur Entfernung mit 1/r^2 abklingen lassen.)

    # Normiere das Ergebnis auf 0-255 und konvertiere zu uint8
    noise_norm = cv2.normalize(noise_blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    noise_map = noise_norm.astype(np.uint8)
    return noise_map