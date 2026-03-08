import gradio as gr
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
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]

    return {labels[i]: float(predictions[i]) for i in range(len(labels))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Coral Reef Species Classifier",
    description="Upload a coral reef image to identify its species"
).launch()