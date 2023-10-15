import tensorflow as tf
from yolov2keras import config
from tensorflow.keras import layers,Model
from .utils import yolo_reshape


def getMobileNet(pretrained=True):
    x_input=layers.Input(shape=(None,None,3))
    x=layers.Lambda(lambda x:x/255.)(x_input)
    x=tf.keras.applications.MobileNet(include_top=False, weights='imagenet' if pretrained else None)(x)
    x=layers.Conv2D((config.num_anchors*(5+len(config.classnames))),(1,1),strides=(1,1),padding='same',name='last_conv')(x)
    out=yolo_reshape(config.num_anchors,(5+len(config.classnames)))(x)
    model=Model(x_input,out,name='yolo_v2_mobilenet')
    return model
