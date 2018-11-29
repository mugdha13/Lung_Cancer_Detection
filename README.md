# Lung_Cancer_Detection

# About
This model is a CNN model that detects solitary pulmonary nodules in CT Scans of Lungs.

# Architecture
The neural network, which has 192*192 input parameters and 70,000,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two fully-connected layers with a final 2-way softmax. To make training faster, we used 10-validation method. To reduce overfitting in the fully-connected layers,a recently-developed regularization method called “dropout” has proven to be very effective.

# Training and Testing

train.py

# CNN Architecture

cnn_model.py

# Utility

util.py
