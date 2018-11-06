import scipy.misc

import tensorflow as tf

import cnn_styletransfer_helpers

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

content_image = scipy.misc.imread("datasets/louvre_small.jpg")
content_image = cnn_styletransfer_helpers.reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("datasets/monet.jpg")
style_image = cnn_styletransfer_helpers.reshape_and_normalize_image(style_image)

generated_image = cnn_styletransfer_helpers.generate_noise_image(content_image)

model = cnn_styletransfer_helpers.load_vgg_model("models/imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = cnn_styletransfer_helpers.compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = cnn_styletransfer_helpers.compute_style_cost(sess, model, STYLE_LAYERS)

J = cnn_styletransfer_helpers.total_cost(J_content, J_style)

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

cnn_styletransfer_helpers.model_nn(sess, generated_image, model, train_step, J, J_content, J_style)