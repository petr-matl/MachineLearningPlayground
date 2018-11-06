import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import math

import nn_tensor_helpers

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = nn_tensor_helpers.load_data()

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

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
X_train = train_x_flatten/255.
X_test = test_x_flatten/255.

print ("train_x's shape: " + str(X_train.shape))
print ("test_x's shape: " + str(X_test.shape))
print('\n')

Y_train = nn_tensor_helpers.convert_to_one_hot(train_y_orig, classes.size)
print ("train_y's shape: " + str(Y_train.shape))
# tf.one_hot(indices=train_y_orig, depth=classes.size, axis=0)
Y_test = nn_tensor_helpers.convert_to_one_hot(test_y_orig, classes.size)
print ("test_y's shape: " + str(Y_test.shape))
# tf.one_hot(indices=train_y_orig, depth=classes.size, axis=0)

print("Grad Optimizer")
_, _, parameters = nn_tensor_helpers.model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, optimizer_engine='grad')

print('-----------------------------')

print("Adam Optimizer")
_, _, parameters = nn_tensor_helpers.model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, optimizer_engine='adam')