import cv2
import numpy as np

def find_aruco_markers(frame, dictionary=cv2.aruco.DICT_4X4_50):
    # Graustufenbild erzeugen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ArUco Wörterbuch laden
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    # Marker erkennen
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        #cv2.imshow("Detected Markers", frame)
        #cv2.waitKey(1)
    else:
        print("Keine Marker gefunden")

    if ids is None or len(ids) < 4:
        print("Nicht genug Marker erkannt.")
        return None

    print(f"Gefundene Marker-IDs: {ids}")

    if ids is None or len(ids) < 4:
        print("Nicht genug Marker erkannt.")
        return None

    # Wir brauchen genau 4 Marker mit IDs 0,1,2,3
    ids = ids.flatten()
    points = []

    for marker_id in [0, 1, 2, 3]:
        if marker_id in ids:
            index = np.where(ids == marker_id)[0][0]
            c = corners[index][0]
            center = c.mean(axis=0)
            points.append(center)
        else:
            print(f"Marker ID {marker_id} nicht gefunden.")
            return None

    points = np.array(points, dtype=np.float32)

    target_points = np.array([
        [0, 0],
        [1280, 0],
        [1280, 960],
        [0, 960]
    ], dtype=np.float32)

    # Homographie berechnen
    H, status = cv2.findHomography(points, target_points)
    return H, points


def apply_homography(frame, H):
    warped = cv2.warpPerspective(frame, H, (1280,960))
    return warped
