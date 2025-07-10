import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model
model = load_model("smile_model.h5")

st.title("Smiling or Not Face Detector")
st.write("Upload a face image and let the model predict if the person is smiling.")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    label = "Smiling" if prediction > 0.5 else "Not Smiling"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
