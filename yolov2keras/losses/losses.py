from yolov2keras import config,utils
import tensorflow as tf
import tensorflow.keras.backend as K



##############################################all losses##################################


def obj_loss(y_true,y_pred):
  y_true_conf=y_true[...,0:1]
  y_pred_conf=K.sigmoid(y_pred[...,0:1])

  y_true_xy=y_true[...,1:3]
  y_pred_xy=K.sigmoid(y_pred[...,1:3])


  y_pred_wh=y_pred[...,3:5]


  obj=y_true[...,0]==1

  box_pred = K.concatenate([y_pred_xy,K.clip(K.exp(y_pred_wh)*config.tf_anchors,0,config.output_size)],axis=-1)
  ious=tf.stop_gradient(utils.GetIoU(y_true[...,1:5][obj],box_pred[obj]))

  obj_loss=tf.reduce_mean(K.square(ious*y_true_conf[obj]-y_pred_conf[obj]))  # y_pred_conf should represent iou

  return obj_loss

def noobj_loss(y_true,y_pred):

  y_true_conf=y_true[...,0:1]
  y_pred_conf=K.sigmoid(y_pred[...,0:1])

  noobj=y_true[...,0]==0

  noobj_loss=tf.reduce_mean(K.binary_crossentropy(y_true_conf[noobj],y_pred_conf[noobj]))
  return noobj_loss

def box_loss(y_true,y_pred):

  y_true_xy=y_true[...,1:3]
  y_pred_xy=K.sigmoid(y_pred[...,1:3])

  y_true_wh=K.log(1e-16+ y_true[...,3:5]/config.tf_anchors)
  y_pred_wh=y_pred[...,3:5]


  obj=y_true[...,0]==1

  box_loss=tf.reduce_mean(K.square(K.concatenate([y_true_xy,y_true_wh],axis=-1)[obj]-K.concatenate([y_pred_xy,y_pred_wh],axis=-1)[obj]))

  return box_loss

def class_loss(y_true,y_pred):

  y_true_class=y_true[...,5:]
  y_pred_class=y_pred[...,5:]

  obj=y_true[...,0]==1

  class_loss=tf.reduce_mean(K.categorical_crossentropy(y_true_class[obj],y_pred_class[obj],from_logits=True))
  return class_loss

default_loss_weights={
                        "obj_loss":1,
                        "noobj_loss":10,
                        "box_loss":10,
                        "class_loss":1
                     }
