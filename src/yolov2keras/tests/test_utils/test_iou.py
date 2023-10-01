# Copyright 2023 The yolov2keras Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from yolov2keras.utils import iou


# TODO(AnujPanthri): Test `iou.GetIoU` for `np.ndarray` here.
def test_iou_numpy():
    true_boxes = np.array([[10, 10, 20, 20],[15, 10, 20, 20], [30, 30, 40, 40],[40, 40, 20, 20]])
    pred_boxes = np.array([[10, 10, 20, 20]])
    iou_tf = iou.GetIoU(true_boxes, pred_boxes)
    
    assert tf.reduce_all(iou_tf==iou.GetIoU(true_boxes,np.tile(pred_boxes,[true_boxes.shape[0],1]))).numpy()  # checking if broadcasting is working
    assert tf.reduce_all(iou_tf==iou.GetIoU(pred_boxes,true_boxes)).numpy()  # checking if order doesn't matters
    assert iou_tf.shape==[true_boxes.shape[0]]
    assert tf.reduce_all(tf.logical_and(iou_tf>=0,iou_tf<=1)).numpy()  # should be between 0 and 1
    assert abs(iou_tf[0]-1)<1e-4  # complete intersection
    assert abs(iou_tf[1]-(300/500))<1e-4  # test 1 intersection
    assert abs(iou_tf[2]-(100/1900))<1e-4  # test 2 intersection
    assert iou_tf[3]<1e-4  # no intersection

# TODO(AnujPanthri): Test `iou.GetIoU` for `tf.Tensor` here.
def test_iou_tf():
    
    true_boxes = K.variable([[10, 10, 20, 20],[15, 10, 20, 20], [30, 30, 40, 40],[40, 40, 20, 20]])
    pred_boxes = K.variable([[10, 10, 20, 20]])
    iou_tf = iou.GetIoU(true_boxes, pred_boxes)
    
    assert tf.reduce_all(iou_tf==iou.GetIoU(true_boxes,tf.tile(pred_boxes,[true_boxes.shape[0],1]))).numpy()  # checking if broadcasting is working
    assert tf.reduce_all(iou_tf==iou.GetIoU(pred_boxes,true_boxes)).numpy()  # checking if order doesn't matters
    assert iou_tf.shape==[true_boxes.shape[0]]
    assert tf.reduce_all(tf.logical_and(iou_tf>=0,iou_tf<=1)).numpy()  # should be between 0 and 1
    assert abs(iou_tf[0]-1)<1e-4  # complete intersection
    assert abs(iou_tf[1]-(300/500))<1e-4  # test 1 intersection
    assert abs(iou_tf[2]-(100/1900))<1e-4  # test 2 intersection
    assert iou_tf[3]<1e-4  # no intersection