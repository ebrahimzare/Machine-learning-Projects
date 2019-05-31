# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:07:14 2019

@author: Ebi
"""

from keras.datasets import mnist

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)

X_train=train_images.reshape(60000, 784)
X_test=test_images.reshape(10000,784)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train /=255
X_test /=255