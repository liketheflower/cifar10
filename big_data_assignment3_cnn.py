
# coding: utf-8

#!/usr/bin/env python2

"""
Created on April 2017

@author: Jimmy Shen
"""

#import six.moves.cPickle as pickle
import cPickle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import keras
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
#from matplotlib import pyplot
from scipy.misc import toimage

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

seed = 7
np.random.seed(seed)

# split the data set into train and test. The test is 10% of the whole data set.
X_train, X_test, y_train, y_test = train_test_split(features, lbl, test_size=0.1, random_state=seed)
num_pixels = X_train.shape[1]

"""
    #Normalize the X_train to 0~1.0
    min_value_X_train=np.nanmin(X_train)
    max_value_X_train=np.nanmax(X_train)
    X_train=(X_train-min_value_X_train)/(max_value_X_train-min_value_X_train)
    
    #Normalize the X_test to 0~1.0
    min_value_X_test=np.nanmin(X_test)
    max_value_X_test=np.nanmax(X_test)
    X_test=(X_train-min_value_X_test)/(max_value_X_test-min_value_X_test)
    

    """
#plot the first 9 images of the train data set.
#print the first image before reshape. The shape of the training data is : 4500X3720
#print("X_train[0] shape before reshape",X_train[0].shape)
#print("X_train[0] before reshape",X_train[0])

X_train = X_train.reshape(X_train.shape[0], 3, 32, 32 )
X_test  = X_test.reshape(X_test.shape[0], 3, 32, 32 )
#print("X_train[0] shape after reshape",X_train[0].shape)
#print("X_train[0] after reshape",X_train[0])
"""for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))
    # show the plot
    pyplot.show()"""
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
print("X_train.shape",X_train.shape)
print("y_train.shape",y_train.shape)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Accuracy: %.2f%%" % (scores[1]*100))
