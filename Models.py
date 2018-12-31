import os
import sys
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import numpy as np
import string

from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import numpy as np
from keras.models import Sequential
from keras.layers.core import  Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.utils import np_utils
#convolutional layers
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.convolutional import MaxPooling2D,MaxPooling1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,AveragePooling1D
from keras.layers import Bidirectional
from keras.models import load_model

from sklearn.model_selection import StratifiedKFold

np.random.seed(1671)
seed = 1671
np.random.seed(seed)

# network and training
NB_EPOCH = 150
BATCH_SIZE = 50
VERBOSE = 1
NB_CLASSES = 2 # number of classes
METRICS =['accuracy']
LOSS = 'binary_crossentropy'
KERNEL_INITIAL ='glorot_uniform'



#stop training if
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint # to save models at each epoch
#Stop training when a monitored quantity has stopped improving.
#patience: number of epochs with no improvement after which training will be stopped.


from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Reshape
from keras.constraints import maxnorm
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report


def SaveHistory(Tuning,outfile):
    Hist = np.empty(shape=(len(Tuning.history['val_loss']),4))
    Hist[:,0] = Tuning.history['val_loss']
    Hist[:,1] = Tuning.history['val_acc']
    Hist[:,2] = Tuning.history['loss']
    Hist[:,3] = Tuning.history['acc']
    np.savetxt(outfile, Hist, fmt='%.8f',delimiter=",",header="val_loss,val_acc,train_loss,train_acc",comments="")
    return Hist

def GetMetrics(model,x,y):
    pred = model.predict_classes(x)
    pred_p=model.predict(x)
    fpr, tpr, thresholdTest = roc_curve(y, pred_p)
    aucv = auc(fpr, tpr) 
    #print('auc:',aucv)
    #print('auc,acc,mcc',aucv,accuracy_score(y,pred),matthews_corrcoef(y,pred))
    precision,recall,fscore,support=precision_recall_fscore_support(y,pred,average='macro')
    #print(classification_report(y,pred))
    #print('mcc:',matthews_corrcoef(y,pred))
    print('auc,acc,mcc,precision,recall,fscore,support:',aucv,accuracy_score(y,pred),matthews_corrcoef(y,pred),precision,recall,fscore,support)
    return [aucv,accuracy_score(y,pred),matthews_corrcoef(y,pred),precision,recall,fscore,support]
    

def one_CNN(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,name):
    
    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_miniCNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH,
                            validation_data= (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])  
    print("train") 
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_miniCNN.txt")
    return Tuning,model


def two_CNN(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,name):
    
    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(MaxPooling1D())
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_smallCNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    #callbacks_list = [checkpoint]
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    #finalmodel = load_model(filepath)
    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_smallCNN.txt")
    return Tuning,model#,roc_train,roc_test,acc_train,acc_test


def three_CNN(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,name):

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_mediumCNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,validation_data= (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_mediumCNN.txt")
    return Tuning,model #,roc_train,roc_test,acc_train,acc_test


def four_CNN(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,name):

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
   
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_largeCNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])

    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_largeCNN.txt")
    return Tuning,model #,roc_train,roc_test,acc_train,acc_test

def eight_CNN(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,name):

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    #model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
   
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())

    

    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_verylargeCNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    #finalmodel = load_model(filepath)
    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_veryLargeCNN.txt")
    return Tuning,model#,roc_train,roc_test,acc_train,acc_test


def one_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,RNN_UNITS,name):
    

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))

    #LSTM
    #HIDDEN_UNITS = 10
    model.add(Bidirectional(LSTM(HIDDEN_UNITS,kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.5)))
    model.add(Flatten())
    model.add(Dense(1))
    
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "_best_miniCNN_RNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH,
                            validation_data= (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])  
    print("train") 
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "_best_miniCNN_RNN.txt")
    return Tuning,model


def two_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,RNN_UNITS,name):
    
    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(MaxPooling1D())
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    #LSTM
    #HIDDEN_UNITS = 10
    model.add(Bidirectional(LSTM(HIDDEN_UNITS,kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.5)))
    model.add(Flatten())
    model.add(Dense(1))
    #a soft max classifier
    model.add(Activation("sigmoid"))
    
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    
    
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "best_smallCNN_RNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "_best_smallCNN_RNN.txt")
    return Tuning,model#,roc_train,roc_test,acc_train,acc_test


def three_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,RNN_UNITS,name):

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    #LSTM
    #HIDDEN_UNITS = 20
    model.add(Bidirectional(LSTM(RNN_UNITS,kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.5)))
    model.add(Flatten())
    model.add(Dense(1))
    #a soft max classifier
    model.add(Activation("sigmoid"))
    
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) +  "best_mediumCNN_RNN.hdf5"
    #filepath = "_".join([name,str(BATCH_SIZE),KERNEL_INITIAL,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) +  "best_mediumCNN_RNN.hdf5"
    #filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_mediumCNN_RNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data =(x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print("train,"+filepath)
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test,"+filepath)
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "_best_mediumCNN_RNN.txt")
    return Tuning,model#,roc_train,roc_test,acc_train,acc_test
    

def four_CNN_LSTM(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,RNN_UNITS,name):

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
   
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())

    #LSTM
    #HIDDEN_UNITS = 10
    model.add(Bidirectional(LSTM(HIDDEN_UNITS,kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.5)))

    
    model.add(Flatten())
    model.add(Dense(1))
    #a soft max classifier
    model.add(Activation("sigmoid"))
    #filepath="smallCNN_dropout_"+str(DROP_OUT)+".hdf5"
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "best_largeCNN_RNN.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    #callbacks_list = [checkpoint]
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    #Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
    #                        validation_split = 0.2) #(x_test,y_test),callbacks=[checkpoint,early_stopping_monitor,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    #finalmodel = load_model(filepath)
    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(RNN_UNITS)]) + "_best_largeCNN_RNN.txt")
    return Tuning,model #,roc_train,roc_test,acc_train,acc_test

def one_CNN_Dense(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,NUM_DENSE_LAYERS,name):
    
    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    for i in range(0,NUM_DENSE_LAYERS):
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
    #LSTM
    #HIDDEN_UNITS = 10
    #model.add(Bidirectional(LSTM(HIDDEN_UNITS,kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.5)))
    #model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(NUM_DENSE_LAYERS)]) + "_best_miniCNN_Dense.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH,
                            validation_data= (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])  
    print("train") 
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL),str(NUM_DENSE_LAYERS)]) + "_best_miniCNN_Dense.txt")
    return Tuning,model

def two_CNN_Dense(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,NUM_DENSE_LAYERS,name):
    
    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(MaxPooling1D())
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    for i in range(0,NUM_DENSE_LAYERS):
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
    model.add(Dense(1))
    #a soft max classifier
    model.add(Activation("sigmoid"))
    
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    
    
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_smallCNN_Dense.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_smallCNN_Dense.txt")
    return Tuning,model#,roc_train,roc_test,acc_train,acc_test

def three_CNN_Dense(x_train,y_train,x_test,y_test,x_val,y_val,learning_rate,INPUT_SHAPE,KERNEL_SIZE,NUM_KERNEL,NUM_DENSE_LAYERS,name):

    model = Sequential()
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL,input_shape = INPUT_SHAPE))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())

    model.add(Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL))
    model.add(Activation("relu"))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    for i in range(0,NUM_DENSE_LAYERS):
        model.add(Dense(100))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
    model.add(Dense(1))
    #a soft max classifier
    model.add(Activation("sigmoid"))
    
    model.compile(loss=LOSS, optimizer = Adam(lr=learning_rate), metrics =METRICS)
    
    
    filepath="_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "best_mediumCNN_Dense.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
    
    print(model.summary())
    Tuning = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                            validation_data = (x_val,y_val),callbacks=[checkpoint,early_stopping_monitor])#,roc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))]) #,callbacks=callbacks_list)

    print("train")
    GetMetrics(load_model(filepath),x_train,y_train)
    print("test")
    GetMetrics(load_model(filepath),x_test,y_test)
    SaveHistory(Tuning,"_".join([name,str(learning_rate),str(KERNEL_SIZE),str(NUM_KERNEL)]) + "_best_mediumCNN_Dense.txt")
    return Tuning,model#,roc_train,roc_test,acc_train,acc_test
    


def SaveHistory(Tuning,outfile):
    #keys = Tunning.history.keys()
    Hist = np.empty(shape=(len(Tuning.history['val_loss']),4))
    Hist[:,0] = Tuning.history['val_loss']
    Hist[:,1] = Tuning.history['val_acc']
    Hist[:,2] = Tuning.history['loss']
    Hist[:,3] = Tuning.history['acc']
    np.savetxt(outfile, Hist, fmt='%.8f',delimiter=",",header="val_loss,val_acc,train_loss,train_acc",comments="")
    return Hist

def SaveResult(roc_train,roc_test,acc_train,acc_test,outfile):
    #keys = Tunning.history.keys()
    f=open(outfile,"w")
    f.write("roc_train,roc_test,acc_train,acc_test"+"\n")
    f.write(str(roc_train) + "," + str(roc_test) + "," + str(acc_train) + "," + str(acc_test))
    f.close()
