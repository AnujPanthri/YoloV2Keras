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

from typing import Union

import numpy as np


# Do we really need to support `np.ndarray` for calculating `IoU` when
# `yolov2keras` uses `tf.Tensor`?
def GetIoU(
  y_true: Union[np.ndarray, tf.Tensor], y_pred: Union[np.ndarray, tf.Tensor]
) -> np.ndarray:
  """Calculate Intersection over Union (IoU) between sets of bounding boxes.

  See the Notes: sections to look at the usage with `tf.Tensor`.

  Args:
    y_true: Ground truth bounding boxes in the format [x, y, w, h].
    y_pred: Predicted bounding boxes in the format [x, y, w, h].

  Returns:
    numpy.ndarray: An array containing IoU values for each pair of boxes.

  Notes:
    - IoU values range from 0 (no overlap) to 1 (perfect overlap).
    - Use `tf.compat.v1.enable_eager_execution()` to enable eager execution.
      Once eager execution is enabled, operations are executed as they are
      defined and Tensor objects hold concrete values, which can be accessed as
      `numpy.ndarray``s through the `numpy()` method.

  Examples:

    Calculate IoU using NumPy arrays:

    >>> true_boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    >>> pred_boxes = np.array([[12, 12, 18, 18], [35, 36, 42, 45]])
    >>> iou_numpy = GetIoU(true_boxes, pred_boxes)
    >>> print(iou_numpy)
    [0.75 0.15]

    Calculate IoU using TensorFlow tensors converted to NumPy arrays:

    >>> import tensorflow as tf
    >>> tf.compat.v1.enable_eager_execution()  # Enable eager execution
    >>> true_boxes_tf = tf.constant([[10, 10, 20, 20], [30, 30, 40, 40]])
    >>> pred_boxes_tf = tf.constant([[12, 12, 18, 18], [35, 36, 42, 45]])
    >>> true_boxes_np = true_boxes_tf.numpy()
    >>> pred_boxes_np = pred_boxes_tf.numpy()
    >>> iou_tf = GetIoU(true_boxes_np, pred_boxes_np)
    >>> print(iou_tf)
    [0.75 0.15]
  """
  box1_x1 = y_true[:, 0:1] - y_true[:, 2:3] / 2
  box1_y1 = y_true[:, 1:2] - y_true[:, 3:4] / 2
  box1_x2 = y_true[:, 0:1] + y_true[:, 2:3] / 2
  box1_y2 = y_true[:, 1:2] + y_true[:, 3:4] / 2

  box2_x1 = y_pred[:, 0:1] - y_pred[:, 2:3] / 2
  box2_y1 = y_pred[:, 1:2] - y_pred[:, 3:4] / 2
  box2_x2 = y_pred[:, 0:1] + y_pred[:, 2:3] / 2
  box2_y2 = y_pred[:, 1:2] + y_pred[:, 3:4] / 2

  xmins = np.clip(np.maximum(box1_x1, box2_x1), 0, None)
  ymins = np.clip(np.maximum(box1_y1, box2_y1), 0, None)
  xmaxs = np.minimum(box1_x2, box2_x2)
  ymaxs = np.minimum(box1_y2, box2_y2)

  intersection = (xmaxs - xmins) * (ymaxs - ymins)
  union = (box1_x2 - box1_x1) * (box1_y2 - box1_y1) + (box2_x2 - box2_x1
                                                       ) * (box2_y2 - box2_y1)
  ious = intersection / (union - intersection + 1e-6)

  return ious
