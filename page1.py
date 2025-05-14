import cv2
import logging
import streamlit as st

from typing import Optional, Dict
from setting import CAMERA
from simple_facerec import SimpleFacerec

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

fr = SimpleFacerec()
fr.load_encoding_images("/home/stechoq/Documents/self-learning/face_recognition/fc/db")
cam = initialize_camera(CAMERA['index'])

st.title("Face Recognition")
placeholderbtn = st.empty()
placeholderimg = st.empty()
placeholdertext = st.empty()

if placeholderbtn.button("Start Capture Image"):
    placeholderbtn.empty()
    try:
        while True:
            ret, frame = cam.read()
            faceLoc, faceName = fr.detect_known_faces(frame)
            for face_loc, name in zip(faceLoc, faceName):
                y, x, h, w = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0), 2)
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

            placeholderimg.image(frame, channels="BGR")
            if faceName:
                name = faceName[0]
                placeholdertext.write(name)
            else:
                name = "Unknown"
                placeholdertext.write("Unknown")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        logger.error(f"Error: {e}")
        pass