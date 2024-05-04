# Neural Network for Iris Classification

## Overview
This assignment involves implementing a multi-layer neural network from scratch using Python to classify Iris plants based on their attributes. The neural network will be trained using the Fisher's Iris dataset

## Requirements
- Python 3
- NumPy
- Dataset
- Sci-Kit Learn
- The Fisher's Iris dataset contains the following attributes:
    - Sepal length (in cm)
    - Sepal width (in cm)
    - Petal length (in cm)
    - Petal width (in cm)
    - Class (Iris Setosa, Iris Versicolour, Iris Virginica)

## The data is split into:
### 50% For Training Set, 25% for Validation Set, %25 for Testing Set
### It's split randomly, so try the program over and over again to see diff. results!

## Functionalities and Architecture
- Training the Neural Network: The neural network is trained using the Fisher's Iris dataset. 
                            Itloads data for training and validation purposes, splitting 
                            it into training and validation sets.

- We train w/ forward + backward propogation, also backtesting with the validation test
to see if we're still getting error loss, the iteration keeps going. If there is no 
error loss on the validation set after 10 iterations, we end early. If the loss keeps
going, then it manually stops after 10k iterations

- Since we have 4 variables for input data, we start a input layer with 4 nodes. Then we create
a hidden layer with double the size, and an output layer that has 3 nodes 
(1 for 3 different classification probabilities)

- I chose this architecture because I've already had some experience with Neural Nets and
Tensorflow, so I decided to create a simple 3 layer model, since the dataset is fairly small

- So, when we finish the output layer, we're given an array of 3 values. Each value is 
the probability of the flower being of which type



