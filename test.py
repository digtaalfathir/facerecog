import streamlit as st
from PIL import Image

st.title("Select Camera and Capture Image")

# Let users capture an image using their camera
image_file = st.camera_input("Take a picture", disabled=False)

if image_file:
    # Convert to PIL Image and display
    image = Image.open(image_file)
    st.image(image, caption="Captured Image")