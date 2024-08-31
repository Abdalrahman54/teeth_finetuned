import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\Abdalrahman\Downloads\best_model.keras")

    # Define a function to preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match your model's input shape
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    st.write(f"Preprocessed image shape: {image.shape}")
    return image

# Define a function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    st.write(f"Model raw prediction: {prediction}")
    return prediction


# Streamlit app interface
st.title("Image Classification with Your Model")
st.write("Upload an image to classify")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    prediction = predict(image)
    class_names = ['Gum', 'OLP', 'OC', 'OT', 'CoS', 'CaS', 'MC']
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")


