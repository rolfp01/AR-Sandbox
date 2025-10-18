import cv2
import numpy as np

def generate_aruco_marker(id, size=200, dictionary=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, size)
    return marker_img

if __name__ == "__main__":
    for marker_id in range(4):
        marker = generate_aruco_marker(marker_id)
        filename = f"aruco_marker_{marker_id}.png"
        cv2.imwrite(filename, marker)
        print(f"Marker {marker_id} gespeichert als {filename}")
