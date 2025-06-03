import gdown
import tensorflow as tf
import os

# Google Drive file ID and output path
GDRIVE_FILE_ID = "1ipZOrL6XvFQj8ibRmVs5MQ7mHI4fOEf-"
MODEL_PATH = "model.h5"

def download_model_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if os.path.exists(output_path):
        print(f"Model already downloaded at {output_path}")
        return
    print("Downloading model from Google Drive...")
    gdown.download(url, output_path, quiet=False)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to download the model to {output_path}")

def load_keras_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == "__main__":
    download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
    model = load_keras_model(MODEL_PATH)

    # Now you can use `model` to predict or whatever your app needs
