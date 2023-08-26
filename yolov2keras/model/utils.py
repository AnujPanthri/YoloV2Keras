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

import tensorflow as tf


@tf.keras.utils.register_keras_serializable('yolov2')
class YoloV2Reshape(tf.keras.layers.Layer):
  def __init__(self, n_anchors, last_item, **kwargs):
    super(YoloV2Reshape, self).__init__(**kwargs)
    self.last_item = last_item
    self.n_anchors = n_anchors

  def call(self, output_layer):
    shape = [tf.shape(output_layer)[k] for k in range(4)]
    return tf.reshape(
      output_layer,
      [shape[0], shape[1], shape[2], self.n_anchors, self.last_item]
    )

  def compute_output_shape(self, input_shape):
    return (
      input_shape[0], input_shape[1], input_shape[2], self.n_anchors,
      self.last_item
    )

  def get_config(self):
    config = super(YoloV2Reshape, self).get_config()
    config.update({'last_item': self.last_item, 'n_anchors': self.n_anchors})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
