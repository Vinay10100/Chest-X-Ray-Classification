# Chest X-Ray Image Classifier
This project aims to classify chest X-ray images into three categories: COVID-19, pneumonia, and normal, using a Convolutional Neural Network (CNN) model.

# Dataset
The dataset used for this project consists of a collection of chest X-ray images obtained from [Chest X-Ray Images Dataset (Kaggle)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset), including COVID-19-positive cases, pneumonia cases, and normal cases.

# Model Architecture
* The first layer starts with 16 filters and a 3x3 kernel.
* The number of filters is doubled at every next layer, and the kernel size is incremented by 1.
* Max Pooling layers are introduced after Convolutional layers to avoid overfitting and reduce computational costs.
* The output from the Convolutional layers is flattened and passed on to Dense layers.
* The first Dense layer starts with 128 neurons and is reduced by half in the next two Dense layers.
* Dropout layers are introduced throughout the model to randomly ignore some neurons and reduce overfitting.
* ReLU activation is used in all layers, except for the output layer, to reduce computation costs and introduce non-linearity.
* The output layer contains 3 neurons, one for each class (Covid, Pneumonia, Normal), with softmax activation.
