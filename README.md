# Yolov2keras

yolov2 implemented in tensorflow keras.


## Supported dataset formats:-

- Pascal Voc

## Models Available:-

- Yolo v2
- Mobilenet


## Train and Save
```py
import yolov2keras as yod 
import tensorflow as tf


train_image_dir="roboflow.voc/train/"
train_annotation_dir="roboflow.voc/train/"

val_image_dir="roboflow.voc/valid/"
val_annotation_dir="roboflow.voc/valid/"

# finding classnames of all the objects in the dataset
classnames_path = yod.dataset.VOCDataset.get_classnames_path(train_annotation_dir,val_annotation_dir)
yod.set_config(input_size=416,num_anchors=5,classnames_path=classnames_path)

# albumentations augmentations for making images a square
train_transform, val_transform, test_transform = yod.dataset.augmentations.default_augmentation()

# returns tf dataset objects
train_ds=yod.ParseDataset(train_image_dir,train_annotation_dir,format="PASCAL_VOC",augment=train_transform)
val_ds=yod.ParseDataset(val_image_dir,val_annotation_dir,format="PASCAL_VOC",augment=val_transform)

# finding the anchors of shape: (n,2)
anchors=yod.dataset.find_anchors(train_ds)
yod.set_anchors(anchors)

# convert to standard format to yolo v2 format
train_ds=yod.yoloDataset(train_ds,batch_size=4,drop_remainder=True)
val_ds=yod.yoloDataset(val_ds,batch_size=4)

# creating the model
# model = yod.models.get_model(basemodel="yolov2",pretrained=True)
model = yod.models.get_model(basemodel="mobilenet",pretrained=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
metrics = [yod.metrics.iou_acc , yod.metrics.class_acc ] + [yod.losses.obj_loss,yod.losses.noobj_loss,yod.losses.box_loss,yod.losses.class_loss]
mapcallback = yod.callbacks.MAPCallback(val_ds,iou_thres=0.5,per_nth_epoch=1)

model.compile(optimizer=optimizer,loss=yod.losses.yolo_loss,metrics=metrics)

model.fit(train_ds,validation_data=val_ds,epochs=5,verbose=1,callbacks=[mapcallback])

# exporting the model
model_path="output/v1/"
yod.save(model_path,model)
```

## Inference

```py
import yolov2keras as yod 
import tensorflow as tf


model_path="output/v1/"

# object_detector = yod.load_model(model_path)
object_detector = yod.load_model_from_weights(model_path)
object_detector.set_config(p_thres=0.5,nms_thres=0.3,image_size=[416])

img="Sample.jpg"

detections = object_detector.predict(img)
print(detections)

yod.inference.helper.show_objects(img,detections)
```

