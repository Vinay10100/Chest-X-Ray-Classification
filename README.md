# Chest X-Ray Image Classifier
This project aims to classify chest X-ray images into three categories: COVID-19, Pneumonia, and Normal, using a Convolutional Neural Network (CNN) model.
The project also includes a deployment using Streamlit for interactive web-based prediction of chest X-ray images.
* Test Images : [Sample Images for Testing Model](https://github.com/Vinay2022/Chest-X-Ray-Classification/tree/main/Test_images)

## Dataset
The dataset used for this project consists of a collection of chest X-ray images obtained from [Chest X-Ray Images Dataset (Kaggle)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset), including COVID-19-positive cases, pneumonia cases, and normal cases.

## Model Architecture
* Convolutional layer 1: 16 filters, 3x3 kernel, ReLU activation, followed by MaxPooling2D layer (2x2 pool size).
* Convolutional layer 2: 64 filters, 3x3 kernel, ReLU activation, padding set to 'same', followed by MaxPooling2D layer (2x2 pool size).
* Dropout layer (0.25) added after Convolutional layer 2.
* Convolutional layer 3: 128 filters, 3x3 kernel, ReLU activation, padding set to 'same', followed by MaxPooling2D layer (2x2 pool size).
* Dropout layer (0.3) added after Convolutional layer 3.
* Convolutional layer 4: 128 filters, 3x3 kernel, ReLU activation, padding set to 'same', followed by MaxPooling2D layer (2x2 pool size).
* Dropout layer (0.4) added after Convolutional layer 4.
* Output from convolutional layers is flattened using a Flatten layer.
* First dense layer: 128 neurons, ReLU activation.
* Dropout layer (0.25) added after the first dense layer.
* Second dense layer: 64 neurons, ReLU activation.
* Output layer: 3 neurons (one for each class), softmax activation for multi-class classification.

## Model Evaluation
![merge_from_ofoct](https://user-images.githubusercontent.com/97530517/231857828-bfd7ce92-2b2c-456f-a339-534a87d8da69.jpg)
![classification](https://user-images.githubusercontent.com/97530517/231857615-47340376-1d2d-4918-b2e6-f141b56273ce.PNG)
![confusion_matrix](https://user-images.githubusercontent.com/97530517/231857703-a2c9aac9-f217-4095-b63d-6145e7b95de8.PNG)

## Sample Predictions
![predict](https://user-images.githubusercontent.com/97530517/231856798-74574e8d-fb31-45b0-a681-b9579900924d.jpg)

## How to Run

To run the Streamlit app, follow these steps:

1. Install the required dependencies, Run the Streamlit app:

   ```bash
   pip install -r requirements.txt

   streamlit run predict.py
3. Upload a chest X-ray image and click the "Predict" button to get the Result.
