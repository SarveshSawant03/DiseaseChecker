import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load models
pneumonia_model = tf.keras.models.load_model("pneumonia_model.h5")
tb_model = tf.keras.models.load_model("tb_model.h5")
skin_model = tf.keras.models.load_model("skin_cancer_model.h5")


def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model's expected input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


st.title("AI-Powered Disease Detection")

body_part = st.selectbox("Select Body Part", ["Chest (TB & Pneumonia)", "Skin (Skin Cancer)"])
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)

    predictions = {}
    if "Chest" in body_part:
        pneumonia_pred = pneumonia_model.predict(processed_image)[0][0]
        tb_pred = tb_model.predict(processed_image)[0][0]
        predictions = {
            "Pneumonia": "Detected" if pneumonia_pred > 0.5 else "Not Detected",
            "TB": "Detected" if tb_pred > 0.5 else "Not Detected"
        }
    elif "Skin" in body_part:
        skin_pred = skin_model.predict(processed_image)[0][0]
        predictions = {"Skin Cancer": "Detected" if skin_pred > 0.5 else "Not Detected"}

    st.subheader("Results")
    for disease, status in predictions.items():
        st.write(f"**{disease}**: {status}")
