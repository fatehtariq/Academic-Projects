# Artificial Neural Networks
###### Semester Project - Handwritten Digits Classification with Deep Convolutional Neural Networks
_________________________________________________________________________
## Introduction
The main goal is to construct and train an artificial neural network on thousands of images of
handwritten digits so that it may successfully identify others when presented. The data that
will be incorporated, is the **MNIST** database which contains 60,000 images for training and
10,000 test images with the help of Keras Python API with TensorFlow as the backend.
Most Handwritten Classification Models depend on a Linear Model in which the matrix data
is first converted into a fixed length vector and then passed on into the Neural Network.
However a better approach would be to use a Convolutional Neural Network and use various
kernels to extract the features.

![image](https://user-images.githubusercontent.com/59331234/118637783-9b917500-b7ef-11eb-8ea4-778ee51b730e.png)

## Methodology
###### Data collection
The MNIST dataset is conveniently bundled within Keras, and we can easily analyze some
of its features in Python.
It contains 28x28 pixel size images of HandWritten Digits with over 60,000 training images
and 10,000 test images.

![image](https://user-images.githubusercontent.com/59331234/118637867-b19f3580-b7ef-11eb-88f6-d37f24464e9c.png)

###### Evaluation Methodology
The Sequential Convolutional Model will consist of 4 Convolutional Layers with 2 Pooling
Layers and 2 Full Connection Layers.

###### Convolution Layer 1
- 32 (3x3) Kernels will be used which will result in 32 feature maps.
- Batch Normalization will be used in this layer
- reLU Activation Function
###### Convolution Layer 2
- 32 (3x3) Kernels will be used which will result in 32 feature maps
- Batch Normalization
- reLU Activation Function
- A 2x2 Max Pooling Layer will be added in the end
###### Convolution Layer 3
- 64 (3x3) Kernels will be used which will result in 64 feature maps
- Batch Normalization will be used in this layer
- reLU Activation Function
###### Convolution Layer 4
- 64 (3x3) Kernels will be used which will result in 64 feature maps
- Batch Normalization will be used in this layer
- reLU Activation Function
- A 2x2 Max Pooling Layer will be added in the end
- And a Flattening Layer will be connected at the Final Convolution Layer

###### Full Connection Layer 1
- 512 Full Connection Nodes will be used
- Batch Normalization
- reLU Activation Function

###### Full Connection Layer 2
- 20% Drop Off rate for 512 Full Connected Nodes
- 10 Final Full Connected Nodes
- A Softmax Layer to produce final output of probabilities

The Model will be evaluated according to the Train and Test Split of the Data
Evaluation Metrics
- Loss Function will be Categorical Cross Entropy
- Adam Optimizer Function will be used
- Metric will be set as accuracy
The Batch Size is kept at 128 ,while the number of steps per epoch is calculated as the total
Sample Size divided by Batch Size, the number of epochs is set at 5. The Validation step
size is kept at 10000 divided by Batch Size. This greatly reduces computational costs in
deeper Convolutional Neural Networks.
The Final Result of the Model concludes an accuracy of over 99.28% with a 0.0223 Loss

## Demonstration of Downloaded Dataset
The MNIST database which contains 60,000 images for training and 10,000 test images with
the help of Keras Python API with TensorFlow as the backend.
In the image above, each row represents one labeled example. Column 0 represents the
label that a human rater has assigned for one handwritten digit. For example, if Column 0
contains '6', then a human rather interpreted the handwritten character as the digit '6'. The
ten digits 0-9 are each represented, with a unique class label for each possible digit. Thus,
this is a multi-class classification problem with 10 classes. Columns 1 through 784 contain
the feature values, one per pixel for the 28×28=784 pixel values. The pixel values are on a
gray scale in which 0 represents white, 255 represents black, and values between 0 and 255
represent shades of gray.

###### Results
###### Training Results
- Final Accuracy: 0.9806
- Loss: 0.0319
###### Test Analysis
- Final Accuracy: 0.9928
- Loss: 0.0223

## Selected Papers
- **Classification of MNIST Handwritten Digit Database using Neural Network** *By
Wan Zhu, Research School of Computer Science, Australian National University,
Acton, ACT 2601, Australia*
- **A Survey of Handwritten Character Recognition with MNIST and EMNIST** *By
Alejandro Baldominos, Computer Science Department, Universidad Carlos III of
Madrid, 28911 Leganés, Madrid, Spain*
