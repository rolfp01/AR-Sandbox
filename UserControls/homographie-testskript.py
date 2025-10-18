import cv2
import numpy as np

# Konfiguration
aruco_dict_type = cv2.aruco.DICT_4X4_50
marker_ids_required = [0, 1, 2, 3]

# Zielpunkte (z.B. Beamerauflösung 1280x960)
target_points = np.array([
    [0, 0],
    [1280, 0],
    [1280, 960],
    [0, 960]
], dtype=np.float32)

def find_marker_centers(corners, ids):
    centers = {}
    for i, id_ in enumerate(ids.flatten()):
        if id_ in marker_ids_required:
            c = corners[i][0]
            center = c.mean(axis=0)
            centers[id_] = center
    return centers

def main():
    # Bild mit Markern laden (z.B. ein Frame aus der Kamera oder gespeichertes Bild)
    image_path = "dein_markerbild.jpg"  # Pfad anpassen
    frame = cv2.imread(image_path)
    if frame is None:
        print("Bild konnte nicht geladen werden:", image_path)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        print("Keine Marker erkannt.")
        return

    centers = find_marker_centers(corners, ids)
    if len(centers) < 4:
        print("Nicht alle benötigten Marker erkannt:", marker_ids_required)
        return

    # Markerzentren in korrekter Reihenfolge (0,1,2,3)
    src_points = np.array([centers[i] for i in marker_ids_required], dtype=np.float32)

    # Homographie berechnen
    H, status = cv2.findHomography(src_points, target_points)
    if H is None:
        print("Homographie konnte nicht berechnet werden.")
        return

    # Bild transformieren
    warped = cv2.warpPerspective(frame, H, (1280, 960))

    # Ursprüngliches Bild mit Markern anzeigen
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("Original mit Markern", frame)

    # Transformiertes Bild anzeigen
    cv2.imshow("Transformiertes Bild (Homographie angewandt)", warped)

    print("Drücke eine Taste zum Beenden.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
