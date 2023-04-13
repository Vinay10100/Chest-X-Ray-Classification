# Chest X-Ray Image Classifier
This project aims to classify chest X-ray images into three categories: COVID-19, pneumonia, and normal, using a Convolutional Neural Network (CNN) model.

## Dataset
The dataset used for this project consists of a collection of chest X-ray images obtained from [Chest X-Ray Images Dataset (Kaggle)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset), including COVID-19-positive cases, pneumonia cases, and normal cases.

## Model Architecture
* The first convolutional layer has 16 filters with a kernel size of 3x3, and ReLU activation function.
* A MaxPooling2D layer with a pool size of 2x2 is added after the first convolutional layer.
The second convolutional layer has 64 filters with a kernel size of 3x3, and ReLU activation function. It also has padding set to 'same' to maintain the spatial dimensions.
* Another MaxPooling2D layer with a pool size of 2x2 is added after the second convolutional layer.
* A Dropout layer with a dropout rate of 0.25 is added after the second MaxPooling2D layer to prevent overfitting.
* The third convolutional layer has 128 filters with a kernel size of 3x3, and ReLU activation function. It also has padding set to 'same'.
* A MaxPooling2D layer with a pool size of 2x2 is added after the third convolutional layer.
* Another Dropout layer with a dropout rate of 0.3 is added after the third MaxPooling2D layer.
* The fourth convolutional layer has 128 filters with a kernel size of 3x3, and ReLU activation function. It also has padding set to 'same'.
* A MaxPooling2D layer with a pool size of 2x2 is added after the fourth convolutional layer.
* Another Dropout layer with a dropout rate of 0.4 is added after the fourth MaxPooling2D layer.
* The output from the convolutional layers is flattened using a Flatten layer to be passed to the dense layers.
* The first dense layer has 128 neurons with ReLU activation function.
* A Dropout layer with a dropout rate of 0.25 is added after the first dense layer.
* The second dense layer has 64 neurons with ReLU activation function.
* The output layer has 3 neurons (one for each class) with a softmax activation function for multi-class classification.
