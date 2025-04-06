import cv2

def picture_In_Gray(cameraInput: cv2.typing.MatLike):
        return cv2.cvtColor(cameraInput, cv2.COLOR_BGR2GRAY)