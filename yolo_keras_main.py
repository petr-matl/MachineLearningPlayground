from keras import backend as K
from keras.models import load_model

import yolo_keras_helpers

sess = K.get_session()

class_names = yolo_keras_helpers.read_classes("datasets/coco_classes.txt")
anchors = yolo_keras_helpers.read_anchors("datasets/yolo_anchors.txt")
image_shape = (720., 1280.)   

yolo_model = load_model("models/yolo.h5")

yolo_outputs = yolo_keras_helpers.yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_keras_helpers.yolo_eval(yolo_outputs, image_shape)

out_scores, out_boxes, out_classes = yolo_keras_helpers.predict(sess, "datasets/test.jpg", scores, boxes, classes, yolo_model, class_names)
