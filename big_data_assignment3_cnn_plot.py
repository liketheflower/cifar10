
# coding: utf-8

#!/usr/bin/env python2

"""
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
from matplotlib import pyplot
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
numpy.random.seed(seed)

# split the data set into train and test. The test is 10% of the whole data set.
X_train, X_test, y_train, y_test = train_test_split(features, lbl, test_size=0.1, random_state=seed)
num_pixels = X_train.shape[1]

#plot the first 9 images of the train data set.
#print the first image before reshape. The shape of the training data is : 4500X3720
print("X_train[0] shape before reshape",X_train[0].shape)
print("X_train[0] before reshape",X_train[0])

X_train = X_train.reshape(X_train.shape[0], 3, 32, 32 )
print("X_train[0] shape after reshape",X_train[0].shape)
print("X_train[0] after reshape",X_train[0])
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()

