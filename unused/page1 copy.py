import cv2
import logging
import streamlit as st
import face_recognition
import os
import numpy as np

from typing import Optional, Dict
from setting import CAMERA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def loadImageDetected(image_path):
    res = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(res)[0]

def loadImageDb(image_path):
    res = face_recognition.load_image_file(image_path)
    return face_recognition.face_encodings(res)[0]

folder_db = f"/home/stechoq/Documents/self-learning/face_recognition/fc/db/"
folder_detected = f"/home/stechoq/Documents/self-learning/face_recognition/fc/detected/"

placeholderbtn = st.empty()
placeholder = st.empty()
placeholderimg = st.empty()
placeholdertxt = st.empty()
if placeholderbtn.button("Start Capture Image"):
    placeholderbtn.empty()
    try:
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise Exception("Failed to initialize camera")

        while True:
            ret, frame = cam.read()
            if not ret:
                logger.error("Failed to get frame")
                break

            placeholder.image(frame, channels="BGR")
            cv2.imwrite(f'{folder_detected}image.png', frame)
            face_detected = face_recognition.load_image_file(f'{folder_detected}image.png')
            face_detected = face_recognition.face_encodings(face_detected)
            face_detected = np.array(face_detected)
            if face_detected is not None:
                for file in os.listdir(folder_db):
                    if file.endswith(".png"):
                        face_db = face_recognition.load_image_file(f'{folder_db}{file}')
                        face_db = face_recognition.face_encodings(face_db)
                        face_db = np.array(face_db)
                        results = face_recognition.compare_faces(face_detected, face_db)
                        print(len(results))
                        if len(results) > 0:
                            if results[0] == True:
                                file.replace(".png", "")
                                print("It's a picture of me!")
                                placeholdertxt.write(f"It's a picture of {file}!")
                        if len(results) == 0:
                            placeholdertxt.write("Face not found")
                        print("hasill", results)
                    else:
                        placeholdertxt.write.info("Face not found")
                        placeholderimg.empty()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Exiting...")
        cam.release()
        cv2.destroyAllWindows()