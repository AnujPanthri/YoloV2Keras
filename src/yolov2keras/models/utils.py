import tensorflow as tf


# tf.keras.saving.get_custom_objects().clear()
@tf.keras.utils.register_keras_serializable("yolov2")  # 1
class yolo_reshape(tf.keras.layers.Layer):
    ''' custom layer for reshaping last layer '''
    def __init__(self, num_anchors, last_item, **kwargs):
        super(yolo_reshape, self).__init__(**kwargs)
        self.last_item = last_item
        self.num_anchors = num_anchors

    def call(self, output_layer):
        shape = [tf.shape(output_layer)[k] for k in range(4)]
        return tf.reshape(
            output_layer,
            [shape[0], shape[1], shape[2], self.num_anchors, self.last_item],
        )

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            self.num_anchors,
            self.last_item,
        )

    def get_config(self):
        config = super(yolo_reshape, self).get_config()
        config.update({"last_item": self.last_item, "num_anchors": self.num_anchors})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
