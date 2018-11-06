import resnet_keras_helpers

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnet_keras_helpers.load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = resnet_keras_helpers.convert_to_one_hot(Y_train_orig, 6).T
Y_test = resnet_keras_helpers.convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model = resnet_keras_helpers.ResNet50(input_shape = (64, 64, 3), classes = 6)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))