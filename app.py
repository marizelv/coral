import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
labels = open("labels.txt").read().splitlines()

IMG_SIZE = 256

def predict(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    return predictions

# Streamlit UI
st.title("Coral Reef Species Classifier")
st.write("Upload a coral reef image to identify its species.")

uploaded_file = st.file_uploader("Choose a coral image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        preds = predict(image)

        # Get top 3 predictions
        top_indices = np.argsort(preds)[::-1][:3]

        st.subheader("Top Predictions")
        for i in top_indices:
            st.write(f"{labels[i]} : {preds[i]*100:.2f}%")
