
# TADBoundaryDectector

Detection of boundaries of TADs (Topologically associated domains) using Deep Learning

## Models.py

Contains the deep learning models as python functions.
Depends on
- TensorFlow
- Keras
- Python >= 3

## Example Sequence Files 

The 'dm3.kc167.example.h5' contains numpy matrices in one-hot encoding format for the training DNA sequences
- training sequences and class labels (0 or 1) x_train and y_train 
- validation sequences and class labels (0 or 1) x_val and y_val
- testing sequences and class labels (0 or 1) x_test and y_test

