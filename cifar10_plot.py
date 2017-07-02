# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage, imresize
#matplotlib.use('GTK')
import numpy as np
# load data

#print(X_train.shape)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print (X_train.shape)
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()
X_train_new = np.zeros((9,224,224,3))
for i in range(9):
    X_train_new[i] = imresize(X_train[i], (224,224,3), interp='bilinear', mode=None)
#X_train_new = imresize(X_train, (X_train.shape[0],224,224,3), interp='bilinear', mode=None)
print(X_train_new.shape) 
#X_train_new = np.zeros((X_train[0].shape))
for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
      #  X_train_new=imresize(X_train[i], (224,224,3), interp='bilinear', mode=None)
        print(X_train_new[i].shape)
        pyplot.imshow(toimage(X_train_new[i]))
# show the plot
pyplot.show()


