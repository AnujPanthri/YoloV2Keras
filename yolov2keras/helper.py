# TODO(AnujPanthri): Replace the imports for `np_get_iou` and `tf_get_iou` with
# the generalised `utils.iou.GetIoU`.

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def np_get_iou(y_true, y_pred):
    """Intersection Over Union checks overlapping between objects.
        * 0 indicates no overlapping.
        * 1 indicates perfect overlapping.

    Args:
        y_true (objects, [x, y, w, h])
        y_pred (objects, [x, y, w, h])
    """
    box1_x1 = y_true[:,0:1] - y_true[:,2:3] / 2
    box1_y1 = y_true[:,1:2] - y_true[:,3:] / 2
    box1_x2 = y_true[:,0:1] + y_true[:,2:3] / 2
    box1_y2 = y_true[:,1:2] + y_true[:,3:] / 2

    box2_x1 = y_pred[:,0:1] - y_pred[:,2:3] / 2
    box2_y1 = y_pred[:,1:2] - y_pred[:,3:] / 2
    box2_x2 = y_pred[:,0:1] + y_pred[:,2:3] / 2
    box2_y2 = y_pred[:,1:2] + y_pred[:,3:] / 2


    xmins = np.maximum(box1_x1,box2_x1)
    ymins = np.maximum(box1_y1,box2_y1)

    xmaxs = np.minimum(box1_x2,box2_x2)
    ymaxs = np.minimum(box1_y2,box2_y2)



    intersection = np.clip((xmaxs-xmins),0,None)*np.clip((ymaxs-ymins),0,None)

    union = (box1_x2-box1_x1)*(box1_y2-box1_y1) + (box2_x2-box2_x1)*(box2_y2-box2_y1)
    ious=intersection/((union-intersection)+1e-6)

    return ious

def tf_get_iou(y_true,y_pred):
  '''
  y_true (objects,[x,y,w,h])
  y_pred (objects,[x,y,w,h])
  '''
  box1_x1 = y_true[:,0:1] - y_true[:,2:3] / 2
  box1_y1 = y_true[:,1:2] - y_true[:,3:4] / 2
  box1_x2 = y_true[:,0:1] + y_true[:,2:3] / 2
  box1_y2 = y_true[:,1:2] + y_true[:,3:4] / 2

  box2_x1 = y_pred[:,0:1] - y_pred[:,2:3] / 2
  box2_y1 = y_pred[:,1:2] - y_pred[:,3:4] / 2
  box2_x2 = y_pred[:,0:1] + y_pred[:,2:3] / 2
  box2_y2 = y_pred[:,1:2] + y_pred[:,3:4] / 2

  x1 = K.max(K.concatenate([box1_x1,box2_x1],axis=-1),axis=-1)
  y1 = K.max(K.concatenate([box1_y1,box2_y1],axis=-1),axis=-1)

  x2 = K.min(K.concatenate([box1_x2,box2_x2],axis=-1),axis=-1)
  y2 = K.min(K.concatenate([box1_y2,box2_y2],axis=-1),axis=-1)

  intersection = K.clip((x2-x1),0,None)*K.clip((y2-y1),0,None)
  intersection = K.expand_dims(intersection ,axis=-1)

  box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
  box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
  # union = (K.abs((box1_x2-box1_x1)*(box1_y2-box1_y1)) + K.abs((box2_x2-box2_x1)*(box2_y2-box2_y1)))
  union = box1_area+box2_area
  ious=intersection/((union-intersection)+1e-6)

  # ious=tf.where(tf.math.is_nan(ious),tf.zeros_like(ious),ious)

  if tf.math.is_nan(tf.reduce_mean(ious)):
    tf.print('please see:')
    nan_idx=tf.math.is_nan(ious)
    tf.print(nan_idx.shape)
    # if nan_idx.shape[0]!=None:
    #   tf.print('box1_area:',y_true[nan_idx])

    tf.print('box1_area:',box1_area[tf.math.is_nan(box1_area)])
    tf.print('box2_area:',box2_area[tf.math.is_nan(box2_area)])

    tf.print('intersection:',intersection[tf.math.is_nan(intersection)])
    tf.print('union:',union[tf.math.is_nan(union)])
    tf.print('nan_ious:',ious[nan_idx])
    # ious=tf.where(nan_idx,tf.zeros_like(ious),ious)
    # tf.print('ious_after_fix:',ious[nan_idx])

  return ious
  # return K.zeros_like(y_true)[:,0:1]
