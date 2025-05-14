import os

root = os.getcwd()

CAMERA = {
    'index': 0,  # Default camera (0 is usually built-in webcam)
    'width': 640,
    'height': 480
}

PATHS = {
    'cascade': f"{root}/haarcascade_frontalface_alt2.xml"
}

# Face detection settings
FACE_DETECTION = {
    'scale_factor': 1.1,
    'min_neighbors': 40,
    'min_size': (40, 40)
}