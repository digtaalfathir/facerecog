import cv2
import logging
import streamlit as st
from typing import Optional
from setting import CAMERA
from simple_facerec import SimpleFacerec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fungsi inisialisasi kamera
def initialize_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

# Inisialisasi Face Recognition
fr = SimpleFacerec()
fr.load_encoding_images("/home/stechoq/Documents/self-learning/face_recognition/fc/db")

# Streamlit UI
st.title("Face Recognition")
run = st.checkbox('Start Camera')  # Checkbox untuk memulai
camera = initialize_camera(CAMERA['index']) if run else None

frame_placeholder = st.empty()
text_placeholder = st.empty()

# Loop yang dikontrol oleh Streamlit (bukan while True yang tidak responsif)
if run and camera:
    stop_button = st.button("Stop Camera")
    while run and not stop_button:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture image")
            break

        faceLoc, faceName = fr.detect_known_faces(frame)
        for face_loc, name in zip(faceLoc, faceName):
            y, x, h, w = face_loc
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

        if faceName:
            text_placeholder.write(f"Detected: {faceName[0]}")
        else:
            text_placeholder.write("Detected: Unknown")

    camera.release()
