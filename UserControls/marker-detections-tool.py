import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Pipeline erstellen
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Pipeline starten
    pipeline.start(config)

    # Erstelle ArUco Detector wie vorher
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    try:
        print("Drücke 'q', um das Programm zu beenden.")

        while True:
            # Frames holen
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Frame in numpy Array umwandeln
            frame = np.asanyarray(color_frame.get_data())

            # Marker erkennen
            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, corner in enumerate(corners):
                    corner = corner[0]
                    cX = int(corner[:, 0].mean())
                    cY = int(corner[:, 1].mean())
                    cv2.putText(frame, f"ID: {ids[i][0]}", (cX - 10, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Intel D415 ArUco Marker Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
