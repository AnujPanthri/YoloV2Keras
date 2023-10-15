import yolov2keras as yod 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

train_image_dir="roboflow.voc/train/"
train_annotation_dir="roboflow.voc/train/"

val_image_dir="roboflow.voc/valid/"
val_annotation_dir="roboflow.voc/valid/"

# Inference
model_path="output/v1/"
# model_path="output/v2/"
# model_path="output/pascal_voc/"
# model_path="output/mobilenet/"


object_detector = yod.load_model(model_path)
object_detector.set_config(p_thres=0.5,nms_thres=0.3,image_size=[416])
yod.set_config(input_size=416,classnames_path=model_path+"classnames.txt")

# print(yod.config.input_size)

train_transform, val_transform, test_transform = yod.dataset.augmentations.default_augmentation()

val_transform = A.Compose([
    A.CenterCrop(yod.config.input_size,yod.config.input_size),
    # A.Resize(yod.config.input_size,yod.config.input_size),
    ], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels'],min_visibility=0.99))


train_ds=yod.ParseDataset(train_image_dir,train_annotation_dir,format="PASCAL_VOC",augment=train_transform)
val_ds=yod.ParseDataset(val_image_dir,val_annotation_dir,format="PASCAL_VOC",augment=val_transform)

# yod.dataset.helper.show_examples(val_ds,num_examples=5)


# convert to standard format to yolo v2 format
train_ds=yod.yoloDataset(train_ds,batch_size=4,drop_remainder=True)
val_ds=yod.yoloDataset(val_ds,batch_size=4)




# y_true,y_pred=yod.callbacks.get_all_data(object_detector.model,train_ds,p_thres=0.4)
y_true,y_pred=yod.callbacks.get_all_data(object_detector.model,val_ds,p_thres=0.01)

print(len(y_true),len(y_pred))

# recalls,precisions=yod.callbacks.calculate_MAP_for_class(y_true,y_pred,class_idx=0,iou_thres=0.5)
# ap=np.trapz(precisions,recalls)

# plt.figure()
# plt.title(f"Recall Precision Curve(AP:{ap})")
# plt.plot(recalls,precisions)
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.show()

print("yod.callbacks.calculate_MAP:",yod.callbacks.calculate_MAP(y_true,y_pred,iou_thres=0.5))