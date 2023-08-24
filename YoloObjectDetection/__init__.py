# set configs:- set image size,num_anchors
# parse dataset format to standard format :- pascal voc
# parse it to yolo v2 format :- 
# choose backbone 
# for each backbone create pre_trained weights loader
# get yolo model
# get yolo loss
# train model
# evalulation


from . import dataset
import numpy as np
import tensorflow as tf

# default configs

input_size=608
num_anchors=4



def set_config(input_size,num_anchors,classnames_path):
    globals()['input_size'] = input_size
    globals()['num_anchors'] = num_anchors

    globals()["classnames"] = open(classnames_path,'r').read().split("\n")
    globals()["class_to_idx"] = {classname:idx for idx,classname in enumerate(globals()["classnames"])}
    globals()["idx_to_class"] = {idx:classname for idx,classname in enumerate(globals()["classnames"])}
    
    globals()['class_colors']={class_name:np.random.rand(3) for class_name in globals()["classnames"]}
    

def yoloDataset(image_dir,annotation_dir,format="PASCAL_VOC",augment=None):

    def add_augment(img,obj_names,objs):
        def f(img,obj_names,objs):

            transformed = augment(image=img, bboxes=np.clip(objs,0.0,1.0), class_labels=obj_names) 

            img=transformed['image']
            objs=np.array(transformed['bboxes'],dtype=np.float32)
            obj_names=np.array(transformed['class_labels'],dtype=np.float32)
            return img,objs,obj_names
        
        img,objs,obj_names=tf.numpy_function(f,[img,obj_names,objs],[tf.uint8,tf.float32,tf.float32])
        return img,obj_names,objs

    format = format.upper()
    allowed_formats = ['PASCAL_VOC']
    if format not in allowed_formats:
        raise ValueError(f"invalid format:{format} , choose one out of {allowed_formats}")



    if format == "PASCAL_VOC":
        ds=dataset.voc_dataset.parse(image_dir=image_dir,annotation_dir=annotation_dir)
   
    if augment:
        ds=ds.map(add_augment,num_parallel_calls=tf.data.AUTOTUNE)

    return ds
