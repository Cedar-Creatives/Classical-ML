import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load your trained MNIST model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("MNIST Digit Classifier")
st.write("Draw a digit (0-9) below:")

# Upload or draw an image
uploaded_file = st.file_uploader("Choose a PNG image", type=["png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((28,28))
    image = ImageOps.invert(image)  # Invert colors if necessary
    image = np.array(image)/255.0
    image = image.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    st.write(f"Predicted Digit: **{digit}**")
