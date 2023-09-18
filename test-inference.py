import yolov2keras as yod 
import tensorflow as tf


# Inference
model_path="output/v1/"
# model_path="output/mobilenet/"


object_detector = yod.load_model(model_path)
object_detector.set_config(p_thres=0.5,nms_thres=0.3,image_size=[416])
img="C:/Users/panth/OneDrive/Pictures/Camera Roll/WIN_20230815_13_49_11_Pro.jpg"

detections = object_detector.predict(img)
print(detections)

yod.inference.helper.show_objects(img,detections)
# print(yod.inference.helper.pred_image(img,detections).shape)