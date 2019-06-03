# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:20:06 2019

@author: Ebi
"""
import keras
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from sklearn.model_selection import train_test_split

from keras.layers import Dense
from keras.models import Sequential
#from sklearn.cross_validation import train_test_split

hoda = scipy.io.loadmat('data\\Data_hoda_full.mat')
print(type(hoda))
print(hoda.keys())
print(type(hoda['Data']))
print(hoda['Data'].shape)
print(type(hoda['labels']))
print(hoda['labels'].shape)
data= hoda['Data'].reshape(-1)
data= hoda['Data'].reshape(-1)
print(data.shape)
labels = hoda['labels'].reshape(-1)
print(labels.shape)

plt.figure(figsize = (5,5))
plt.imshow(data[4])
plt.show()

print(labels[4])

for i in range(1,6):
    print(data[i].shape)
    
data_resized = np.array([cv2.resize(img, dsize=(5, 5)) for img in data])    
for i in range(1,6):
    print(data_resized[i].shape)
    
data_norm = data_resized/255
print(data_norm[1])

data_norm = data_norm.reshape(60000,25)
X_train, X_test, y_train, y_test = train_test_split(data_norm,labels)

n_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, n_classes)
y_test_cat = keras.utils.to_categorical(y_test,n_classes)
print(y_train_cat[0])
    
model = Sequential()
model.add(Dense(50,activation = 'relu', input_shape = (25,)))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train_cat, batch_size = 512, epochs=10,verbose = 1)

model.evaluate(X_test,y_test_cat)
preds = model.predict_classes(X_test)
print('y_test:')
print(y_test)
print('preds:')
print(preds)

