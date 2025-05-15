import os
import cv2
import logging
import numpy as np
from PIL import Image
from typing import Optional, Dict
from simple_facerec import SimpleFacerec
from fastapi.responses import JSONResponse
from setting import PATHS, FACE_DETECTION, CAMERA
from fastapi import FastAPI, UploadFile, File, HTTPException, Form



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Inisialisasi face recognizer
fr = SimpleFacerec()
fr.load_encoding_images("/home/stechoq/Documents/self-learning/face_recognition/fc/db")

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

@app.get("/")
def root():
    return {"message": "Face Recognition API is running."}

@app.get("/metrics")
def metrics():
    return {"message": "Metrics endpoint not implemented"}

@app.get("/detect")
def detect_face():
    camera = initialize_camera(CAMERA['index'])
    if not camera:
        return JSONResponse(status_code=500, content={"error": "Failed to initialize camera"})

    # Ambil beberapa frame untuk menstabilkan kamera
    frame = None
    for _ in range(10):  # Bisa disesuaikan, misalnya 20 untuk hasil lebih stabil
        ret, temp_frame = camera.read()
        if ret:
            frame = temp_frame

    camera.release()

    if frame is None:
        return JSONResponse(status_code=500, content={"error": "Failed to capture image"})

    # Deteksi wajah
    faceLoc, faceName = fr.detect_known_faces(frame)

    results = []
    for loc, name in zip(faceLoc, faceName):
        y, x, h, w = loc
        results.append({
            "name": str(name),
            "location": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        })

    return {"detected_faces": results} if results else {
        "detected_faces": [{"name": "Unknown", "location": None}]
    }

@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG or PNG supported.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    faceLoc, faceName = fr.detect_known_faces(image)

    results = []
    for loc, name in zip(faceLoc, faceName):
        y, x, h, w = loc
        results.append({
            "name": str(name),
            "location": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        })

    if results:
        return {"detected_faces": results}
    else:
        return {"detected_faces": [{"name": "Unknown", "location": None}]}

@app.post("/register_face")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG or PNG supported.")

    # Baca file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(PATHS['cascade'])

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION['scale_factor'],
        minNeighbors=FACE_DETECTION['min_neighbors'],
        minSize=FACE_DETECTION['min_size']
    )

    if len(faces) == 0:
        return JSONResponse(status_code=400, content={"message": "No face detected"})

    # Ambil wajah pertama yang terdeteksi dan potong
    for (x, y, w, h) in faces:
        margin = 30
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(image.shape[1], x + w + margin)
        y_end = min(image.shape[0], y + h + margin)

        face_img = image[y_start:y_end, x_start:x_end]
        break  # hanya ambil wajah pertama

    # Simpan wajah ke direktori database
    save_path = os.path.join("/home/stechoq/Documents/self-learning/face_recognition/fc/db/", f"{name}.png")
    success = cv2.imwrite(save_path, face_img)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save face image")
    
    fr.load_encoding_images("/home/stechoq/Documents/self-learning/face_recognition/fc/db")

    return {"message": "Face registered successfully", "name": name}

