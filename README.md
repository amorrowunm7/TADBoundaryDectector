
# TADBoundaryDectector

Detection of boundaries of TADs (Topologically associated domains) using Deep Learning

## Models.py

Contains the deep learning models as python functions.
Depends on
- TensorFlow
- Keras
- Python >= 3


## dm3.kc167.example.h5

The 'dm3.kc167.example.h5' contains numpy matrices in one-hot encoding format for the training DNA sequences
- training sequences and class labels (0 or 1) x_train and y_train 
- validation sequences and class labels (0 or 1) x_val and y_val
- testing sequences and class labels (0 or 1) x_test and y_test

x_train, x_val, and x_test are 3-dimensional matrices\
y_train, y_val, and y_test are 1-dimensional matrices.

## Loading the training, testing, and validation data

import h5py \
filename = 'dm3.kc167.example.h5' \
f = h5py.File(filename, 'r') \
x_train = np.array(f['x_train']) \
x_test = np.array(f['x_test']) \
x_val = np.array(f['x_val']) \
y_train = np.array(f['y_train']) \
y_test = np.array(f['y_test']) \
y_val= np.array(f['y_val']) \

## Setting some parameters
INPUT_SHAPE = x_train.shape[1:3] \
KERNEL_SIZE = 9 \
LEARNING_RATE = 0.001 \
NUM_KERNEL = 64 \
outputFile = 'dm3.kc167' \

## Training CNN models

import Models\
Models.one_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.two_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.three_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.four_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.eight_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
