import YoloObjectDetection as yod 



train_image_dir="roboflow.voc/train/"
train_annotation_dir="roboflow.voc/train/"

val_image_dir="roboflow.voc/valid/"
val_annotation_dir="roboflow.voc/valid/"

classnames_path = yod.dataset.VOCDataset.get_classnames_path(train_annotation_dir)
print(classnames_path)

yod.set_config(input_size=416,num_anchors=5,classnames_path=classnames_path)

# print(yod.classnames)
# print(yod.class_to_idx)
# print(yod.idx_to_class)

train_transform, val_transform, test_transform = yod.dataset.augmentations.default_augmentation()

train_ds=yod.ParseDataset(train_image_dir,train_annotation_dir,format="PASCAL_VOC",augment=train_transform)
val_ds=yod.ParseDataset(val_image_dir,val_annotation_dir,format="PASCAL_VOC",augment=val_transform)



# print(train_ds)


for data in train_ds.take(1):
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)


yod.dataset.helper.show_examples(train_ds,num_examples=5)

anchors=yod.dataset.find_anchors(train_ds)
yod.dataset.helper.show_anchors(anchors)

yod.set_anchors(anchors)

print(yod.anchors)

# convert to standard format to yolo v2 format
# yod.yoloDataset(train_ds)
# print(train_ds)