import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

import cnn_tensor_helpers

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = cnn_tensor_helpers.load_data()

#index = 82
#plt.imshow(train_x_orig[index])
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture."

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y_orig.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y_orig.shape))

conv_layers = {}

X_train = train_x_orig/255.
X_test = test_x_orig/255.

Y_train = cnn_tensor_helpers.convert_to_one_hot(train_y_orig, classes.size).T 
# tf.one_hot(indices=train_y_orig, depth=classes.size, axis=0)
Y_test = cnn_tensor_helpers.convert_to_one_hot(test_y_orig, classes.size).T 
# tf.one_hot(indices=train_y_orig, depth=classes.size, axis=0)

print("Grad Optimizer")
_, _, parameters = cnn_tensor_helpers.model(train_x_orig, Y_train, test_x_orig, Y_test, optimizer_engine='grad')

print('-----------------------------')

print("Adam Optimizer")
_, _, parameters = cnn_tensor_helpers.model(train_x_orig, Y_train, test_x_orig, Y_test, optimizer_engine='adam')