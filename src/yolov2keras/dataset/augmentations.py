import random
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import cv2
import numpy as np
from yolov2keras import config

class BBoxSafeRandomSquareCrop(DualTransform):
    def __init__(self,scale=[1,1.5],p=None):
      self.scale=scale
      self.p = np.array(p) if p!=None else np.array([1 for _ in range(len(scale))])
      self.p = self.p/sum(self.p)

    def __call__(self, image, bboxes, class_labels,**kwargs):

      image_h,image_w=image.shape[:2]

      objs=np.array(bboxes)[:,:-1] # xmin,ymin,xmax,ymax
      objs_w=(objs[:,2]-objs[:,0])*image_w
      objs_h=(objs[:,3]-objs[:,1])*image_h

      selected_idx=np.random.choice(np.arange(0,objs.shape[0]))
      selected_box=objs[selected_idx].copy()
      # print(bboxes)
      # print(selected_idx,selected_box)

      # scale=random.uniform(self.scale[0],self.scale[1])
      scale=np.random.choice(self.scale,p=self.p)
      # print(scale)

      # crop_w=crop_h=min(min(image_w,image_h),np.r_[objs_w,objs_h].max()*scale) # maximum of object widths and heights
      crop_w=crop_h=min(min(image_w,image_h), max(objs_w[selected_idx],objs_h[selected_idx])*scale ) # maximum of selected object width and height

      crop_w=crop_w/image_w # scaling
      crop_h=crop_h/image_h # scaling




      boxes_xmin,boxes_ymin=selected_box[2]-crop_w,selected_box[3]-crop_h
      boxes_xmax,boxes_ymax=selected_box[0],selected_box[1]
      # print("crop_w:",crop_w)

      # print("boxes_xmin:",boxes_xmin,"boxes_xmax:",boxes_xmax)

      boxes_xmin,boxes_ymin=max(0,boxes_xmin),max(0,boxes_ymin)
      boxes_xmax,boxes_ymax=min(boxes_xmax,1-crop_w),min(boxes_ymax,1-crop_h)

      # print("boxes_xmin:",boxes_xmin,"boxes_xmax:",boxes_xmax)
      # print()

      rand_xmin,rand_ymin=random.uniform(boxes_xmin,boxes_xmax),random.uniform(boxes_ymin,boxes_ymax)


      aug=A.Crop(x_min=int(rand_xmin*image_w)            , y_min=int(rand_ymin*image_h),
                 x_max=int((rand_xmin+crop_w)*image_w), y_max=int((rand_ymin+crop_h)*image_h),
                 always_apply=True, p=1)
      X=aug(image=image,bboxes=bboxes,class_labels=class_labels)

      return X





def default_augmentation():
    train_transform=A.Compose([
                            # BBoxSafeRandomSquareCrop(scale=[5,8]),
                            BBoxSafeRandomSquareCrop(scale=[3,5,6,8],p=[1.5,1.5,2,2]),
                            A.Resize(config.input_size,config.input_size),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.GaussNoise(var_limit=(10.0, 50.0)),
                            A.GaussianBlur(blur_limit=(3, 5)),

    # ], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels'],min_visibility=0.90))
    ], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels'],min_visibility=0.99))
    val_transform=A.Compose([
                            BBoxSafeRandomSquareCrop(scale=[5,8]),
                            A.Resize(config.input_size,config.input_size),
    # ], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels'],min_visibility=0.90))
    ], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels'],min_visibility=0.99))

    # TODO: Make available a test set to test our model accuracy.
    test_transform = A.Compose([
    # A.CenterCrop(),
    A.Resize(config.input_size,config.input_size),
    ])

    return train_transform,val_transform,test_transform