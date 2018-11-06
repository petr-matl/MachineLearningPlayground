import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

import nn_helpers

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = nn_helpers.load_data()

#index = 82
#plt.imshow(train_x_orig[index])
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

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

### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
lambdAs = [0, 0.01, 0.03, 0.06, 0.1, 0.33, 0.66, 1, 3, 6, 10]

#for lambd in lambdAs:
    #print('lambda {0}'.format(lambd))
'''
start = datetime.now()
print("Time start {0}".format(start))
parameters = nn.L_layer_model(X_train, train_y_orig, layers_dims, optimizer='gd', num_epochs = 2500, lambd=0.33, mini_batch_size = m_train, print_cost = True)
end = datetime.now()
print("Time end {0}".format(end))
delta = end - start
print("{0} min, {1} sec".format(math.floor(delta.seconds / 60), delta.seconds % 60))
pred_train = nn.predict(X_train, train_y_orig, parameters)
print(pred_train)
pred_test = nn.predict(X_test, test_y_orig, parameters)
print(pred_test)
'''

'''
print("Grad Optimizer")
parameters = nn.L_layer_model(X_train, train_y_orig, layers_dims, optimizer='grad', learning_rate=0.009, num_epochs = 100, print_cost = True)
pred_train = nn.predict(X_train, train_y_orig, parameters)
pred_test = nn.predict(X_test, test_y_orig, parameters)

print('-----------------------------')
'''

print("Adam Optimizer")
parameters = nn_helpers.L_layer_model(X_train, train_y_orig, layers_dims, optimizer='adam', learning_rate=0.009, num_epochs = 100, print_cost = True)
pred_train = nn_helpers.predict(X_train, train_y_orig, parameters)
pred_test = nn_helpers.predict(X_test, test_y_orig, parameters)
