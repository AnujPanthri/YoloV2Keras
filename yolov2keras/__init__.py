# set configs:- set image size,num_anchors
# parse dataset format to standard format :- pascal voc
# parse it to yolo v2 format :- 
# choose backbone 
# for each backbone create pre_trained weights loader
# get yolo model
# get yolo loss
# train model
# evalulation


from . import utils
from . import dataset
from . import models
from . import losses
from . import metrics
from . import config
from . import inference

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math,os,shutil

__name__ = 'yolov2keras'


def set_config(classnames_path,input_size=None,num_anchors=None):
    if input_size:
        if input_size % config.cell_size != 0:
            raise ValueError("INPUT_SIZE_ERROR: choose a input_size which is divisible by {:d}.".format(config.cell_size))

        config.input_size = input_size
        config.output_size = input_size / config.cell_size
    
    if num_anchors: config.num_anchors = num_anchors

    config.classnames = open(classnames_path,'r').read().split("\n")
    config.class_to_idx = {classname:idx for idx,classname in enumerate(config.classnames)}
    config.idx_to_class = {idx:classname for idx,classname in enumerate(config.classnames)}
    config.class_colors = {class_name:np.random.rand(3) for class_name in config.classnames}
    
def set_anchors(anchors):
    config.num_anchors = len(anchors)
    config.anchors = anchors
    config.tf_anchors = K.reshape(K.variable(anchors),[1, 1, 1, config.num_anchors , 2])

def ParseDataset(image_dir,annotation_dir,format="PASCAL_VOC",augment=None,shuffle=False):

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
        ds=dataset.VOCDataset.parse(image_dir=image_dir,annotation_dir=annotation_dir,shuffle=shuffle)
   
    if augment:
        ds=ds.map(add_augment,num_parallel_calls=tf.data.AUTOTUNE)

    return ds


def yoloDataset(ds,batch_size=1,prefetch=True,cache=False,drop_remainder=False):


    xywh_anchors=np.c_[np.zeros_like(config.anchors),config.anchors]  # adding x=0,y=0 to anchors

    def to_yolo_labels(img,obj_names,objs):
        def f(img,obj_names,objs,output_size):
            output_size=int(output_size)
            cell_size=(config.input_size/output_size)

            grid=np.zeros([output_size,output_size,config.num_anchors,1+4+len(config.classnames)])
 
            if len(objs)==0: return grid.astype(np.float32)
 
            #objs are in yolo format center_x,center_y,center_w,center_h
            objs*=output_size # to rescale cordinates between 0 to 13  if(output_size==13)

            for i,obj in enumerate(objs):  # center_x,center_y,center_w,center_h

                obj_r,obj_c=int(np.clip(math.floor(obj[1]),0,output_size-1)),int(np.clip(math.floor(obj[0]),0,output_size-1)) # y,x
                # print(obj_r,obj_c)
               
                obj_to_check=np.r_[np.zeros(2),obj[2:]][None] # adding x,y=0,0
                
                ious=utils.GetIoU(obj_to_check,xywh_anchors)
                best_anchor_idx,best_iou=np.argmax(ious),np.max(ious)
                # print('best_anchor_idx:',best_anchor_idx,'ious:',best_iou)
                
                if (grid[obj_r,obj_c,best_anchor_idx,0]==0):
                    class_one_hot=tf.keras.utils.to_categorical(obj_names[i],num_classes=len(config.classnames))
                    grid[obj_r,obj_c,best_anchor_idx]=[ 1 ,obj[0]-(obj_c) , obj[1]-(obj_r) , obj[2] , obj[3] , *class_one_hot ] # p,x,y,w,h,c_1,c_2...c_n
                    # del class_one_hot

                # print(grid[obj_r,obj_c,anchor_idx])
            return grid.astype(np.float32)
        label=tf.numpy_function(f,[img,obj_names,objs,config.output_size],tf.float32)
        img.set_shape(img.shape)
        label.set_shape([int(config.output_size),int(config.output_size),config.num_anchors,1+4+len(config.classnames)])
        return img,label
    
    ds=ds.map(to_yolo_labels,num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size:ds=ds.batch(batch_size,drop_remainder=drop_remainder)

    if prefetch:ds=ds.prefetch(tf.data.AUTOTUNE)
    if cache:ds=ds.cache()
    return ds

def save(export_dir,model):
    
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    np.savetxt(export_dir+"anchors.txt",config.anchors)
    with open(export_dir+"classnames.txt","w") as f:
        f.write("\n".join(config.classnames))
    model.compile()
    model.save(export_dir+"model.h5")
    return export_dir



def load_model(model_dir):

    set_config(classnames_path=model_dir+"classnames.txt")
    set_anchors(np.loadtxt(model_dir+"anchors.txt"))
    object_detector = inference.Detector(model_dir+"model.h5")

    return object_detector