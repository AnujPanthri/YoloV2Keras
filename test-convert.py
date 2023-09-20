import yolov2keras as yod 
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


model_dir="output/v1/"

yod.set_config(classnames_path=model_dir+"classnames.txt")
yod.set_anchors(np.loadtxt(model_dir+"anchors.txt"))
model = yod.models.getYolov2(pretrained=False)
model.load_weights(model_dir+"model.h5")
model.save(model_dir+"model.h5")

# object_detector = yod.load_model(model_dir)