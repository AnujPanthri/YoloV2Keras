import tensorflow as tf
from tensorflow.keras import Model,layers
from yolov2keras.models.utils import yolo_reshape


def load_model(path):
    model=tf.keras.models.load_model(path,compile=False,custom_objects={"yolo_reshape":yolo_reshape})
    return model
