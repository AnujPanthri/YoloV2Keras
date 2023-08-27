import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import os
import yolov2keras as yod
from abc import ABC , abstractmethod  # for making abstract class methods


class BaseDataset(ABC):
  @abstractmethod
  def get_classnames_path(self):
    pass
  @abstractmethod
  def parse(self):
    pass

  
class VOCDataset(BaseDataset):

  def load_label(xml_file,image_dir):

    objs_list=[]
    tree=ET.parse(xml_file)
    root=tree.getroot()
    width,height=int(root.find('size').find('width').text),int(root.find('size').find('height').text)

    filename=image_dir.decode()+'/'+root.find("filename").text
    x_scale=1/width # to normalize cordinates between 0 and 1
    y_scale=1/height # to normalize cordinates between 0 and 1
    for member in root.findall("object"):
      bndbox=member.find('bndbox')
      xmin,ymin , xmax,ymax=np.clip(float(bndbox.find('xmin').text),0,width),np.clip(float(bndbox.find('ymin').text),0,height),np.clip(float(bndbox.find('xmax').text),0,width),np.clip(float(bndbox.find('ymax').text),0,height)
      value=(
              yod.class_to_idx[member.find('name').text.lower() if len(yod.classnames)>1 else yod.classnames[0]],   # if classnames are more than one then read from xml file otherwise use the classname from classnames list
              # class_to_idx[object_name],              # class
              # xmin,  # xmin
              # ymin,  # ymin

              # xmax,  # xmax
              # ymax,  # ymax

            (xmin+(xmax-xmin)/2) *x_scale,  # center_x
            (ymin+(ymax-ymin)/2) *y_scale,  # center_y

            (xmax-xmin) *x_scale,  # width
            (ymax-ymin) *y_scale,  # height
            )
      objs_list.append(value)
    objs_list=np.array(objs_list,dtype=np.float32)
    # print("see:",objs_list.shape)
    return filename,objs_list


  @classmethod
  def load_data(cls,image_dir):
    def f(annotation_path):
      img_path,objs=tf.numpy_function(cls.load_label,[annotation_path,image_dir],[tf.string,tf.float32])
      img = tf.io.read_file(img_path)
      img = tf.image.decode_image(img,expand_animations=False,channels=3)
      # img = tf.image.decode_jpeg(img)
      objs = tf.cast(objs,dtype=tf.float32)

      return img,objs[:,0],objs[:,1:]
    return f


  @classmethod
  def parse(cls,image_dir,annotation_dir,shuffle):
      annotations_list = glob(annotation_dir+"*.xml")
      dataset = tf.data.Dataset.from_tensor_slices((annotations_list))
      if shuffle: dataset=dataset.shuffle(2048)
      dataset = dataset.map(cls.load_data(image_dir),num_parallel_calls=tf.data.AUTOTUNE)
      return dataset









  @classmethod
  def get_classnames_path(self,*annotation_dirs):
    classnames_path = annotation_dirs[0].rsplit("/")[0]+"/classnames.txt"
    classnames = []
    if not os.path.exists(classnames_path):
      for annotation_dir in annotation_dirs:
        classnames.extend(self.get_class_names(annotation_dir))
      
      with open(classnames_path,"w") as f:
        f.write("\n".join(classnames))
    return classnames_path

  def get_class_names(
          xmls_dir: str
          ) -> list:

    xml_fpaths = glob(xmls_dir+"*.xml")

    annot_objs = []
    annot_cls_names = []
    for fpath in xml_fpaths:
        tree = ET.parse(fpath)
        root = tree.getroot()

        for o in root.findall('object'):
            annot_cls_names = [
                *annot_cls_names,
                o.find('name').text.lower()
            ]          

    annot_cls_names = list(set(annot_cls_names))
    annot_cls_names.sort()

    return annot_cls_names

