import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import os
import requests

# Function to download model from Google Drive
def download_model_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error("Failed to download model from Google Drive.")
        st.stop()

# Constants
MODEL_PATH = "emotion_classifier_inception.h5"
GDRIVE_FILE_ID = "1ipZOrL6XvFQj8ibRmVs5MQ7mHI4fOEf-"  
IMG_HEIGHT, IMG_WIDTH = 224, 224
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocess and prediction
def preprocess_image(image: Image.Image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_emotion(image: Image.Image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    return class_labels[predicted_class]

# Video frame processing callback for webrtc
def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    label = predict_emotion(pil_img)

    # Put label text on frame
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition System")
st.markdown("### Detect the emotion from an image using a trained InceptionV3 model.")

tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label = predict_emotion(image)
        st.markdown(f"### Prediction: `{label}`")

with tab2:
    st.info("Webcam runs via browser, works on Streamlit Cloud and locally.")
    webrtc_streamer(key="emotion-recognizer", video_frame_callback=video_frame_callback)
