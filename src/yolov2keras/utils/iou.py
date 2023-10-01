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
import tensorflow as tf
import tensorflow.keras.backend as K


# Do we really need to support `np.ndarray` for calculating `IoU` when
# `yolov2keras` uses `tf.Tensor`?
def GetIoU(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculate Intersection over Union (IoU) between sets of bounding boxes.

      See the Notes: sections to look at the usage with `tf.Tensor`.

      Args:
        y_true: Ground truth bounding boxes in the format [center_x, center_y, w, h].
        y_pred: Predicted bounding boxes in the format [center_x, center_y, w, h].

      Returns:
        numpy.ndarray: An array containing IoU values for each pair of boxes.

      Notes:
        - IoU values range from 0 (no overlap) to 1 (perfect overlap).
        - Use `tf.compat.v1.enable_eager_execution()` to enable eager execution.
          Once eager execution is enabled, operations are executed as they are
          defined and Tensor objects hold concrete values, which can be accessed as
          `numpy.ndarray``s through the `numpy()` method.

      Examples:

        Calculate IoU using TensorFlow tensors:

        >>> import tensorflow.keras.backend as K
        >>> true_boxes = K.variable([[12, 10, 20, 20], [30, 30, 40, 40]])
        >>> pred_boxes = K.variable([[12, 12, 20, 20]])
        >>> iou_tf = GetIoU(true_boxes, pred_boxes)
        >>> print(iou_tf)
        [0.8181818 0.0775862]

        Calculate IoU using TensorFlow tensors converted to NumPy arrays:
    """
    box1_x1 = y_true[:, 0] - y_true[:, 2] / 2
    box1_y1 = y_true[:, 1] - y_true[:, 3] / 2
    box1_x2 = y_true[:, 0] + y_true[:, 2] / 2
    box1_y2 = y_true[:, 1] + y_true[:, 3] / 2

    box2_x1 = y_pred[:, 0] - y_pred[:, 2] / 2
    box2_y1 = y_pred[:, 1] - y_pred[:, 3] / 2
    box2_x2 = y_pred[:, 0] + y_pred[:, 2] / 2
    box2_y2 = y_pred[:, 1] + y_pred[:, 3] / 2

    xmins = K.maximum(box1_x1, box2_x1)
    ymins = K.maximum(box1_y1, box2_y1)
    xmaxs = K.minimum(box1_x2, box2_x2)
    ymaxs = K.minimum(box1_y2, box2_y2)

    intersection = K.clip((xmaxs - xmins), 0, None) * K.clip(
        (ymaxs - ymins), 0, None)
    union = (box1_x2 - box1_x1) * (box1_y2 - box1_y1) + (box2_x2 - box2_x1) * (
        box2_y2 - box2_y1)
    ious = intersection / (union - intersection + 1e-6)

    return ious
