import YoloObjectDetection as yod 



train_image_dir="roboflow.voc/train/"
train_annotation_dir="roboflow.voc/train/"

val_image_dir="roboflow.voc/valid/"
val_annotation_dir="roboflow.voc/valid/"

classnames_path = yod.dataset.voc_dataset.get_class_names_path(train_annotation_dir)
print(classnames_path)

yod.set_config(input_size=416,num_anchors=5,classnames_path=classnames_path)

# print(yod.classnames)
# print(yod.class_to_idx)
# print(yod.idx_to_class)

train_transform, val_transform, test_transform = yod.dataset.augmentations.default_augmentation()

train_ds=yod.yoloDataset(train_image_dir,train_annotation_dir,format="PASCAL_VOC",augment=train_transform)
val_ds=yod.yoloDataset(val_image_dir,val_annotation_dir,format="PASCAL_VOC",augment=val_transform)



# print(train_ds)


for data in train_ds.take(1):
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)


yod.dataset.helper.show_examples(train_ds,num_examples=5)