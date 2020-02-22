#Feed Forward Neural Network

import tensorflow as tf                                                         #Import needed libraries
from tensorflow import keras                                                    #Machine learning
import numpy as np                                                              #Array handling
import matplotlib.pyplot as plt                                                 #Plotting
import socket                                                                   #UDP Communication
import time
import re
from random_gen import *

examples = 1000

X = random_x(examples,3)
Y = random_ydisc(examples)
Xtrain = X[0:(X.shape[0]*.7),:]
Ytrain = Y[0:(Y.shape[0]*.7),:]
Xtest = X[(X.shape[0]*.7):,:]
Ytest = Y[(Y.shape[0]*.7):,:]

print(X)
print(Y)

"""
index = int(round(a.shape[0] * .7)) #Divide set into training (70%) and test (30%)

Y = a[0:index,10] #Create Y label vector for training.
Y = Y - 1 #Adjust Y label vector values to fit NN.
X = a[0:index,0:6] #Create X matrix for training.
Xtest = a[(index + 1):a.shape[0],0:6] #Create X matrix for testing.
Ytest = a[(index + 1):a.shape[0],10] #Create Y vector for testing.
Ytest = Ytest - 1 #Adjust values of Y vector.

class_names = ['N', 'HH', 'LH', 'LL', 'HL'] #label to know different classes of affetive states within circumplex model of affect.

model = keras.Sequential([ #Declare a secuential Feed Forward Neural Network With Keras.

    keras.layers.Dense(200,input_dim = 6 , activation = 'relu'), #input layer for the model. Takes input with six variables coming from terMITe. Adjust input_dim to add more sensors.
    #Hidden layers sequence. Each layer has 200 neurons with activation fucntion relu on every one of them .
    keras.layers.Dense(200, activation=tf.nn.relu, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    keras.layers.Dense(200, activation=tf.nn.relu, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    keras.layers.Dense(200, activation=tf.nn.relu, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),
    #Output layer has 5 neurons for each one of the five affective states. Output vector contains probabilities of classification.
    keras.layers.Dense(5, activation=tf.nn.softmax, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

])

model.compile(optimizer='rmsprop', #Uses root mean squared error for optimization.
              loss='sparse_categorical_crossentropy', # soarse categorical cross entropy is used as loss function.
              metrics=['accuracy'])

history = model.fit(X, Y, validation_split = 0.33, batch_size = 500, epochs=1500) #Epochs 60, 1000 Training can be done with different combinations of epochs depending on the data set used.
print(history.history.keys()) #terminal outout of accuracy results.

test_loss, test_acc = model.evaluate(Xtest, Ytest) #Evaluate model with test sets (X and Y).

print('Test accuracy:', test_acc) #Terminal print of final accuracy of model.

predictions = model.predict(Xtest) #Uses test set to predict.

model.summary()
model.get_config()
print ('Number of Training Examples Used: ' , Y.size) #Helps get number of training examples used.
print ('Hours of Data;' , (Y.size * 1.5) / 3600) #Calculates hours of data. Intervals of 1.5 seconds are used to obtain data.

#plt.style.use('dark_background')

#Complete sript for plotting end results for accuracy on test and training set across different epochs.
plt.rcParams.update({'font.size': 25})
plt.figure(1)
plt.plot(history.history['acc'], '-') #Plot Accuracy Curve
plt.plot(history.history['val_acc'], ':')
plt.title('Model Accuracy U6')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='lower right')
plt.show()

"""

#plt.figure(2)
#plt.plot(history.history['loss']) #Plot Loss Curvecompletedata = []
#plt.title('Model Loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Test Set'], loc='upper left')
#plt.show()
#plt.figure(3)
#for times in range(100):
#    if np.argmax(predictions[times]) == Ytest[times]:
#        plt.plot(times, (Y[times]), 'go')
#    else:
#        plt.plot(times, np.argmax(predictions[times]), 'rx')
#        plt.plot(times, ((Ytest[times])), 'bo')
#    plt.axis([0, 100, -1, 5])
#    plt.title('Prediction Space')
    #plt.legend()
#plt.show()
