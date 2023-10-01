import yolov2keras as yod 
import tensorflow as tf


train_image_dir="roboflow.voc/train/"
train_annotation_dir="roboflow.voc/train/"

val_image_dir="roboflow.voc/valid/"
val_annotation_dir="roboflow.voc/valid/"

classnames_path = yod.dataset.VOCDataset.get_classnames_path(train_annotation_dir,val_annotation_dir)
print(classnames_path)

# overwrite classnames 
classname='face'
with open(classnames_path,'w') as f:
    f.write(classname)

yod.set_config(input_size=416,num_anchors=5,classnames_path=classnames_path)

# print(yod.config.classnames)
# print(yod.config.class_to_idx)
# print(yod.config.idx_to_class)

train_transform, val_transform, test_transform = yod.dataset.augmentations.default_augmentation()

train_ds=yod.ParseDataset(train_image_dir,train_annotation_dir,format="PASCAL_VOC",augment=train_transform)
val_ds=yod.ParseDataset(val_image_dir,val_annotation_dir,format="PASCAL_VOC",augment=val_transform)



# print(train_ds)


for data in train_ds.take(1):
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)


# yod.dataset.helper.show_examples(train_ds,num_examples=5)

anchors=yod.dataset.find_anchors(train_ds)
# yod.dataset.helper.show_anchors(anchors)

yod.set_anchors(anchors)

print(yod.config.anchors)

# convert to standard format to yolo v2 format
train_ds=yod.yoloDataset(train_ds,batch_size=4,drop_remainder=True)
val_ds=yod.yoloDataset(val_ds,batch_size=4)

for data in val_ds.take(1):
    print(data[0].shape)
    print(data[1].shape)

# yod.dataset.helper.show_yolo_examples(train_ds)

# model = yod.models.getYolov2(pretrained=True)
model = yod.models.getMobileNet(pretrained=True)
# model.summary()

# losses = [yod.losses.obj_loss,yod.losses.noobj_loss,yod.losses.box_loss,yod.losses.class_loss]
# loss_weights = [yod.losses.default_loss_weights[loss.__name__] for loss in losses]
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
metrics = [yod.metrics.iou_acc , yod.metrics.class_acc ] + [yod.losses.obj_loss,yod.losses.noobj_loss,yod.losses.box_loss,yod.losses.class_loss]
# model.compile(optimizer=optimizer,loss=losses,metrics=metrics,loss_weights=loss_weights)
model.compile(optimizer=optimizer,loss=yod.losses.yolo_loss,metrics=metrics)

model.fit(train_ds,validation_data=val_ds,epochs=5,verbose=1)


model_path="output/mobilenet/"
yod.save(model_path,model)