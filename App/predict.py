#importing libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Loading the tflite model
model_path = "main.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']

# Define class names
class_names = ['Covid', 'Viral Pneumonia', 'Normal']

st.set_page_config(page_title="Chest X-ray Classifier", layout="wide")
col1, col2 = st.columns([1, 1])  # Divide the page into two columns

# Left column - Image upload
with col1:
    st.title('Chest X-Ray Classification')
    st.markdown('<h3 style="font-weight:normal;">Classify Chest X-ray images into COVID-19, Pneumonia, or Normal.</h3>', unsafe_allow_html=True)

    # Image upload
    uploaded_file = st.file_uploader("Upload an image and click the 'Predict Now' button.", type=["jpg", "jpeg", "png"])

    # Image prediction
    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")  # Convert to RGB if not already
        image = image.resize((input_shape[1], input_shape[2]))
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)

        # Predict function
        def predict(image):
            interpreter.set_tensor(input_details[0]['index'], image.astype(input_dtype))
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_class_index = np.argmax(predictions, axis=1)
            predicted_class_name = class_names[predicted_class_index[0]]
            return predicted_class_name

        # Predict Now button
        if st.button('Predict Now'):
            predicted_class_name = predict(image)
            # Display prediction as a heading in bold font
            st.markdown(f"<h2>Classified as: <span style='font-style: italic; font-weight: bold;'>{predicted_class_name}</span></h2>", unsafe_allow_html=True)

# Display uploaded image
with col2:
    if uploaded_file is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
