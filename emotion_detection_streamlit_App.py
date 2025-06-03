import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import requests
import time
from io import BytesIO
import gdown

# Constants
FILE_ID = "1-W3XEcLKsce_ULy6BMgEkFdqxIIUoOxk"
MODEL_PATH = "emotion_classifier_inception.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
IMG_HEIGHT, IMG_WIDTH = 150,150
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---- IMAGE PREPROCESSING ----
def preprocess_image(image: Image.Image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_emotion(image: Image.Image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    label = class_labels[predicted_class]
    return label

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition System")
st.markdown("### Detect the emotion from an image using a trained InceptionV3 model.")

tab1, tab2 = st.tabs(["Upload Image", "Use Webcam (Local Only)"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        label = predict_emotion(image)
        st.markdown(f"### Prediction: `{label}`")

with tab2:
    st.warning(" Webcam access works only in local mode. Streamlit Cloud does not support it.")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Could not access webcam.")
        else:
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                label = predict_emotion(pil_img)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                FRAME_WINDOW.image(frame, channels="BGR")
            camera.release()
            cv2.destroyAllWindows()
