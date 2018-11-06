import numpy as np
import tensorflow as tf
from keras import backend as K
K.set_image_data_format('channels_first')
from datetime import datetime
import math

import facenet_keras_helpers
import facenet_inception_v2

start = datetime.now()
print("Load model start: {0}".format(start))
FRmodel = facenet_inception_v2.faceRecoModel(input_shape=(3, 96, 96))
end = datetime.now()
print("Load model end: {0}".format(end))
delta = end - start
print("Load model duration: {0} min, {1} sec".format(math.floor(delta.seconds / 60), delta.seconds % 60))

print("Total Params:", FRmodel.count_params())
FRmodel.compile(optimizer = 'adam', loss = facenet_keras_helpers.triplet_loss, metrics = ['accuracy'])

start = datetime.now()
print("Load weights start: {0}".format(start))
facenet_keras_helpers.load_weights_from_FaceNet(FRmodel)
end = datetime.now()
print("Load weights end: {0}".format(end))
delta = end - start
print("Load weights duration: {0} min, {1} sec".format(math.floor(delta.seconds / 60), delta.seconds % 60))

database = {}
database["danielle"] = facenet_keras_helpers.img_to_encoding("datasets/danielle.png", FRmodel)
database["younes"] = facenet_keras_helpers.img_to_encoding("datasets/younes.jpg", FRmodel)
database["tian"] = facenet_keras_helpers.img_to_encoding("datasets/tian.jpg", FRmodel)
database["andrew"] = facenet_keras_helpers.img_to_encoding("datasets/andrew.jpg", FRmodel)
database["kian"] = facenet_keras_helpers.img_to_encoding("datasets/kian.jpg", FRmodel)
database["dan"] = facenet_keras_helpers.img_to_encoding("datasets/dan.jpg", FRmodel)
database["sebastiano"] = facenet_keras_helpers.img_to_encoding("datasets/sebastiano.jpg", FRmodel)
database["bertrand"] = facenet_keras_helpers.img_to_encoding("datasets/bertrand.jpg", FRmodel)
database["kevin"] = facenet_keras_helpers.img_to_encoding("datasets/kevin.jpg", FRmodel)
database["felix"] = facenet_keras_helpers.img_to_encoding("datasets/felix.jpg", FRmodel)
database["benoit"] = facenet_keras_helpers.img_to_encoding("datasets/benoit.jpg", FRmodel)
database["arnaud"] = facenet_keras_helpers.img_to_encoding("datasets/arnaud.jpg", FRmodel)

facenet_keras_helpers.verify("datasets/camera_0.jpg", "younes", database, FRmodel)

facenet_keras_helpers.verify("datasets/camera_2.jpg", "kian", database, FRmodel)

facenet_keras_helpers.who_is_it("datasets/camera_0.jpg", database, FRmodel)