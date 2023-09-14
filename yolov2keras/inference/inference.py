import cv2
import numpy as np
import math
from yolov2keras import config
import tensorflow.keras.backend as K

from .load_model import load_model
from .decode_yolo_v2 import *


class SquareCrop:

    def __call__(self, img):
        h, w = img.shape[:2]
        self.w_removed, self.h_removed = 0, 0
        if w > h:
            self.w_removed = (w - h) // 2
            img = img[:, self.w_removed:w - self.w_removed]
        elif h > w:
            self.h_removed = (h - w) // 2
            img = img[self.h_removed:h - self.h_removed, :]

        h, w = img.shape[:2]
        self.w_removed, self.h_removed = self.w_removed / w, self.h_removed / h
        return img

    def rescale(self, objs_found):
        raise NotImplementedError


class SquarePad:

    def __init__(self, color=(0, 0, 0)):
        self.color = color

    def __call__(self, img):

        h, w = img.shape[:2]
        self.w_added, self.h_added = 0, 0
        if h > w:
            self.w_added = int((h - w) / 2)
            padding = (np.ones([h, self.w_added, 3]) *
                       np.array(self.color)[None, None, :]).astype("uint8")
            img = np.concatenate([padding, img, padding], axis=1)
        elif w > h:
            self.h_added = int((w - h) / 2)
            padding = (np.ones([self.h_added, w, 3]) *
                       np.array(self.color)[None, None, :]).astype("uint8")
            img = np.concatenate([padding, img, padding], axis=0)
        h, w = img.shape[:2]

        self.w_added, self.h_added = self.w_added / w, self.h_added / h

        return img

    def rescale(self, objs_found):

        for i in range(len(objs_found)):
            objs_found[i][2] = (objs_found[i][2] -
                                self.w_added) / (1 - 2 * self.w_added)
            objs_found[i][3] = (objs_found[i][3] -
                                self.h_added) / (1 - 2 * self.h_added)
            objs_found[i][4] = (objs_found[i][4]) / (1 - 2 * self.w_added)
            objs_found[i][5] = (objs_found[i][5]) / (1 - 2 * self.h_added)
        return objs_found


class Detector:

    def __init__(self, model_path):

        self.model = load_model(model_path)
        self.modes_available = ["sized"]
        self.square_preprocessing = SquarePad()

    def set_config(self,
                   p_thres,
                   nms_thres,
                   batch_size=1,
                   mode="sized",
                   **kwargs):
        # mode : sized

        self.p_thres = p_thres
        self.nms_thres = nms_thres
        self.batch_size = batch_size

        if mode == "sized":
            try:

                self.image_size = kwargs['image_size'] if (
                    type(kwargs['image_size']) == type(list([1])) or
                    type(kwargs['image_size']) == type(np.zeros([]))) else [
                        kwargs['image_size']
                    ]

            except:
                raise ValueError(f"Not all Sized mode parameters passed.")
        else:
            raise ValueError(
                f"Unavailable mode={mode} \nmode can only be one of:{self.modes_available}"
            )

        self.mode = mode

    def invoke_model(self, img):
        all_objs_found = []

        for i in range(math.ceil(img.shape[0] / self.batch_size)):

            y_pred = self.model.predict(img[int(i * self.batch_size):int(
                (i + 1) * self.batch_size)].astype('float32'),
                                        verbose=0)
            # print(y_pred.shape)

            for i in range(y_pred.shape[0]):
                objs_found = get_objects(y_pred[i], p=self.p_thres)
                all_objs_found.append(objs_found)

        return all_objs_found

    def predict_once(self, img, image_size):

        if not hasattr(self, 'mode'):
            raise ValueError(
                f"First call set_config function to set mode using one of the following mode :{self.modes_available}"
            )

        if self.mode == 'sized':
            resized_img = cv2.resize(img, [image_size, image_size])
            objs_found = self.invoke_model(resized_img[None, :, :, :])[0]

        return objs_found

    def predict(self, img):

        if (type(img) == str):
            img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        elif (type(img) != type(np.zeros([]))):
            raise TypeError(f"Inappropriate type of image={type(img)}")

        img = self.square_preprocessing(img)

        if not hasattr(self, 'mode'):
            raise ValueError(
                f"First call set_mode function to set mode using one of the following mode :{self.modes_available}"
            )

        elif self.mode == "sized":
            all_objs_found = []
            for image_size in self.image_size:
                objs_found = self.predict_once(img, image_size)
                all_objs_found.extend(objs_found)

        all_objs_found = np.array(all_objs_found)
      
        all_objs_found = nms(all_objs_found, self.nms_thres)     
        all_objs_found = self.square_preprocessing.rescale(
            all_objs_found)  #rescale coordinates to original image's resolution
        for obj_found in all_objs_found:
            obj_found[1] = config.idx_to_class[obj_found[1]]
        # print(all_objs_found)

        return all_objs_found
