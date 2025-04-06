import cv2

def read_Laptop_Camera():
    camera = cv2.VideoCapture(0) # 0 für die Standardkamera
    while True:
        success, frame = camera.read() # Bild von der Kamera lesen
        if not success:
            break
        else:
            global cameraInput
            cameraInput = frame
            return cameraInput