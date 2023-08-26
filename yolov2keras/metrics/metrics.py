import tensorflow as tf
import tensorflow.keras.backend as K
import yolov2keras as yod

def iou_acc(y_true,y_pred):

  y_pred_xy=K.sigmoid(y_pred[...,1:3])
  y_pred_wh=y_pred[...,3:5]

  obj=y_true[...,0]==1


  box_pred = K.concatenate([y_pred_xy,K.clip(K.exp(y_pred_wh)*yod.tf_anchors,0,yod.output_size)],axis=-1)
  ious=yod.helper.tf_get_iou(y_true[...,1:5][obj],box_pred[obj])

  return ious*100
  # return tf.reduce_mean(ious)

def class_acc(y_true,y_pred):
  obj=y_true[...,0]==1
  y_true_class=K.argmax(y_true[obj][...,5:],axis=-1)
  y_pred_class=K.argmax(K.softmax(y_pred[obj][...,5:]),axis=-1)
  return tf.reduce_mean(tf.cast(y_pred_class==y_true_class,dtype=tf.float32))*100

