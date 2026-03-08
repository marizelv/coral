import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# --------------------------
# Load model and labels
# --------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_data
def load_labels():
    return open("label.txt").read().splitlines()

model = load_model()
labels = load_labels()

IMG_SIZE = 256

# --------------------------
# Prediction function
# --------------------------
def predict(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)[0]
    return predictions

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Coral Reef Species Classifier", layout="wide")

st.title("🐠 Coral Reef Species Classifier")
st.write("Upload a coral reef image to identify its species.")

# Sidebar options
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of Top Predictions", min_value=1, max_value=5, value=3)
show_chart = st.sidebar.checkbox("Show Prediction Confidence Chart", value=True)

# File uploader
uploaded_file = st.file_uploader("Choose a coral image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            preds = predict(image)
        
        # Get top K predictions
        top_indices = np.argsort(preds)[::-1][:top_k]
        top_labels = [labels[i] for i in top_indices]
        top_probs = [preds[i] for i in top_indices]
        
        # Display top predictions
        st.subheader("Top Predictions")
        for lbl, prob in zip(top_labels, top_probs):
            st.markdown(f"**{lbl}** : {prob*100:.2f}%")
        
        # Optional: display bar chart
        if show_chart:
            df = pd.DataFrame({"Species": top_labels, "Confidence": top_probs})
            st.bar_chart(df.set_index("Species"))
