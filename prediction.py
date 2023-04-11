import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import os

# Global variables
covid_interpreter = None
labels = {0: "COVID-19", 1: "Viral Pneumonia", 2: "Normal"}
input_details, output_details = None, None

# Load the model
def load_covid_classifier():
    global covid_interpreter, input_details, output_details
    covid_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.getcwd(), 'model.tflite'))
    covid_interpreter.allocate_tensors()
    input_details, output_details = covid_interpreter.get_input_details(), covid_interpreter.get_output_details()

# Define function to predict image
def predict(image):
    global covid_interpreter, labels
    image = image.convert("RGB").resize((256, 256))
    img = np.array(image, dtype='float32') / 255
    img = img.reshape((1, 256, 256, 3))
    covid_interpreter.set_tensor(input_details[0]['index'], img)
    covid_interpreter.invoke()
    predictions = covid_interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    result = {
        'class': labels[pred],
        'confidence': confidence,
        'confidence_values': predictions[0]
    }
    return result

# Define Streamlit app
def app():
    global covid_interpreter, labels, input_details, output_details
    if covid_interpreter is None:
        load_covid_classifier()

    # Set app title and layout
    st.set_page_config(page_title='Chest X-ray Classifier', page_icon=':microbe:', layout='wide', initial_sidebar_state='auto')
    
    # Add app header
    st.title('Chest X-ray Classifier')
    st.markdown('<h3 style="font-weight:normal;">Classify Chest X-ray images into COVID-19, Viral Pneumonia, or Normal.</h3>', unsafe_allow_html=True)
  
    # Add file uploader
    file_uploaded = st.file_uploader("Upload File", type=['jpg', 'jpeg', 'png'])
    
    # Check if file is uploaded
    if file_uploaded is not None:
        # Check file size
        if file_uploaded.size > (1024 * 1024):
            st.error("File size is too large. Please upload a smaller file.")
            return
        image = Image.open(file_uploaded)

    # Add prediction button
    predict_now = st.button("Predict Now")
    if predict_now and file_uploaded is not None:
        col1, col2 = st.columns([3, 2])
        col2.image(image, width=300, caption="Uploaded Image")
        col1.header("Diagnosis: ")
        result = predict(image)
        col1.write("The X-ray is classified as: **" + result['class'] + "**")
        col1.write("Confidence: **{:.2f}%**".format(result['confidence']))

# Run the Streamlit app
if __name__ == '__main__':
    app()
