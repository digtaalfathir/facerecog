import streamlit as st
import numpy as np
from PIL import Image
import cv2
import logging
from setting import PATHS, FACE_DETECTION, CAMERA
from typing import Optional, Dict


if "btnShowCam" not in st.session_state:
    st.session_state.btnShowCam = False


faceName = st.text_input("Enter your name")
face_cascade = cv2.CascadeClassifier(PATHS['cascade'])
placeholder = st.empty()
placeholderbtn = st.empty()
placholdercamera = st.empty()
btnShowCam = placeholderbtn.button("Start Capture Image")

if btnShowCam:
    st.session_state.btnShowCam = True

if st.session_state.btnShowCam:
    camera_image = placholdercamera.camera_input("Take a picture")
    if faceName == "":
        faceName = "Unknown"
        placholdercamera.empty()
        st.error("Please enter your name")
    
    if camera_image is not None:
        st.session_state.btnShowCam = False
        try:
            image = Image.open(camera_image)
            
            image_array = np.array(image)
            
            img_path = "/home/stechoq/Documents/self-learning/face_recognition/fc/db/"
            
            # Image.fromarray(image_array).save(f"{img_path}{faceName}.png")
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            if len(faces) > 0:
                st.write("Face Detected")
            else:
                st.error("No face detected")

            for (x, y, w, h) in faces:
                # cv2.rectangle(image_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
                margin = 30

                x_start = max(0, x - margin)  
                y_start = max(0, y - margin)  
                x_end = min(image_array.shape[1], x + w + margin)  
                y_end = min(image_array.shape[0], y + h + margin)  

                # face_img = image_array[y_start:y_end, x_start:x_end]
                face_img = image_array[y:y+h, x:x+w]
                
                face_pil = Image.fromarray(face_img)
                face_pil.save(f"{img_path}{faceName}.png")

                face_array = np.array(face_pil)
                gray = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            st.error(f"An error occurred: {e}")