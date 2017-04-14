
# coding: utf-8

#!/usr/bin/env python2

"""
    three layer neural network is used here
Created on April 2017

@author: Jimmy Shen
"""

#import six.moves.cPickle as pickle
import cPickle
from sklearn.model_selection import train_test_split
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import keras


f = open('/Users/jimmy/Dropbox/python_code/big_data/cifar.pkl', 'rb')
dict = cPickle.load(f)
f.close()
print("type of dict",type(dict))
features = dict['data']
print("type of features",type(features))
print("shape of features",features.shape)


lbl = dict['labels']
print("lbl[0]",lbl[0])
print("type of lbl",type(lbl))
print("shape of lbl",lbl.shape)




# added by jimmy shen on Dec 11 2016
#import tensorflow as tf
#import random

seed = 7
numpy.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(features, lbl, test_size=0.1, random_state=seed)
num_pixels = X_train.shape[1]
#y_train = keras.utils.to_categorical(y_train, num_classes=10)
#y_test = keras.utils.to_categorical(y_test, num_classes=10)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Three layer Neural Network Accuracy: %.2f%%" % (scores[1]*100))
