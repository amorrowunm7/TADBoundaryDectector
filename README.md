

# TADBoundaryDectector

Detection of boundaries of TADs (Topologically associated domains) using Deep Learning

## Models.py

Contains the deep learning models as python functions.
Depends on
- TensorFlow
- Keras
- Python >= 3

Contains the following model architectures \

| Python function | Model architecture |
| --------------- | ------------------                   |
| one_CNN         | 1 x {CNN, Maxpooling}                |
| two_CNN         | 2 x {CNN, Maxpooling}                |
| three_CNN       |  3 1 x {CNN, Maxpooling}                |
| four_CNN         | 4 x {CNN, Maxpooling}                |
| eight_CNN         | 2xCNN, {CNN,pooling}, 2xCNN, {CNN,pooling}, CNN, {CNN,pooling}             |
| one_CNN_LSTM       | 1 x {CNN, Maxpooling},1 bidirectional LSTM layer                |
| two_CNN_LSTM         | 2 x {CNN, Maxpooling}, 1 bidirectional LSTM layer               |
| three_CNN_LSTM       |  3 1 x {CNN, Maxpooling}, 1 bidirectional LSTM layer               |
| four_CNN_LSTM         | 4 x {CNN, Maxpooling}, 1 bidirectional LSTM layer                |

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
y_val= np.array(f['y_val']) 

## Setting some parameters
INPUT_SHAPE = x_train.shape[1:3] \
KERNEL_SIZE = 9 \
LEARNING_RATE = 0.001 \
NUM_KERNEL = 64 \
outputFile = 'dm3.kc167' 

## Training CNN models

- Note that training each model will take a couple of hours.
- It will automatically generate a 'hdf5' file that will store the best model, and a '.txt' file that contains the optimization history.
- It will also print out evaluation metrics on the training and testing data using the best model.

import Models\
Models.one_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.two_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.three_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.four_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)\
Models.eight_CNN(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,outputFile)

## Training CNN_LSTM models

- Note that training each model will take a couple of hours.
- It will automatically generate a 'hdf5' file that will store the best model, and a '.txt' file that contains the optimization history.
- It will also print out evaluation metrics on the training and testing data using the best model.

LSTM_UNITS = 40
import Models\
Models.one_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,LSTM_UNITS,outputFile)
Models.two_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,LSTM_UNITS,outputFile)
Models.three_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,LSTM_UNITS,outputFile)
Models.four_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,LSTM_UNITS,outputFile)

## Training CNN_Dense models

- Note that training each model will take a couple of hours.
- It will automatically generate a 'hdf5' file that will store the best model, and a '.txt' file that contains the optimization history.
- It will also print out evaluation metrics on the training and testing data using the best model.

NUM_DENSE_LAYERS=2 \
import Models \
Models.one_CNN_Dense(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,NUM_DENSE_LAYERS,outputFile)

Models.two_CNN_Dense(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,NUM_DENSE_LAYERS,outputFile)

Models.three_CNN_Dense(x_train,y_train,x_test,y_test,x_val,y_val,LEARNING_RATE,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,NUM_DENSE_LAYERS,outputFile)


