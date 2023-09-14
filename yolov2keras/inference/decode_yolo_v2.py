import numpy as np
import tensorflow.keras.backend as K
from yolov2keras import config


def get_objects(y_pred, p=0.5, decode_preds=True):
    '''
      Used to process yolov2 output to objects found list
    '''

    output_size = y_pred.shape[1]

    if decode_preds:
        y_pred[..., 0] = K.sigmoid(y_pred[..., 0])
        y_pred[..., 3:5] = np.clip(
            (K.exp(y_pred[..., 3:5]) * config.tf_anchors).numpy(), 0, output_size)

    objs_found = []
    idxs = np.where(y_pred[..., 0] >= p)
    if np.size(idxs):
        for i, obj in enumerate(y_pred[idxs[0], idxs[1], idxs[2], :]):
            # obj (p,x,y,w,h,c_1,c_2,c_3,c_4,c_5.......c_n)

            if decode_preds:
                obj[1:3] = K.sigmoid(obj[1:3])  # x,y

            prob = obj[0]
            obj = obj[1:]

            obj[4] = np.argmax(obj[4:])
            obj = obj[:5]

            obj[0] = idxs[1][i] + obj[0]  # center x
            obj[1] = idxs[0][i] + obj[1]  # center y

            obj[0] = np.clip(obj[0] - (obj[2] / 2), 0, output_size)  # xmin
            obj[1] = np.clip(obj[1] - (obj[3] / 2), 0, output_size)  # ymin

            obj_details = [prob, obj[4], *list(obj[:-1] / output_size)
                          ]  # xywh are scaled 0 to 1 [P,C_IDX,X,Y,W,H]

            objs_found.append(obj_details)

    return np.array(objs_found)


def list_get_iou(bboxes1, bboxes2):
  # bboxes has xywh => xmin,ymin,width,height
  bboxes1 = [
      bboxes1[0], bboxes1[1], bboxes1[0] + bboxes1[2], bboxes1[1] + bboxes1[3]
  ]
  bboxes2 = [
      bboxes2[0], bboxes2[1], bboxes2[0] + bboxes2[2], bboxes2[1] + bboxes2[3]
  ]

  xA = max(bboxes1[0], bboxes2[0])
  yA = max(bboxes1[1], bboxes2[1])
  xB = min(bboxes1[2], bboxes2[2])
  yB = min(bboxes1[3], bboxes2[3])

  intersection_area = max(0, xB - xA) * max(0, yB - yA)

  box1_area = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
  box2_area = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])

  iou = intersection_area / float(box1_area + box2_area - intersection_area +
                                  1e-6)

  return iou


def np_iou(bboxes1, bboxes2):
  # bboxes has xywh => xmin,ymin,width,height

  boxes1_x1 = bboxes1[:, 0]
  boxes1_y1 = bboxes1[:, 1]
  boxes1_x2 = boxes1_x1 + bboxes1[:, 2]
  boxes1_y2 = boxes1_y1 + bboxes1[:, 3]

  boxes2_x1 = bboxes2[:, 0]
  boxes2_y1 = bboxes2[:, 1]
  boxes2_x2 = boxes2_x1 + bboxes2[:, 2]
  boxes2_y2 = boxes2_y1 + bboxes2[:, 3]

  xmins = np.maximum(boxes1_x1, boxes2_x1)
  ymins = np.maximum(boxes1_y1, boxes2_y1)

  xmaxs = np.minimum(boxes1_x2, boxes2_x2)
  ymaxs = np.minimum(boxes1_y2, boxes2_y2)

  intersection = np.clip((xmaxs - xmins), 0, None) * np.clip(
      (ymaxs - ymins), 0, None)

  union = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1) + (
      boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
  ious = intersection / ((union - intersection) + 1e-6)

  return ious


def nms(objs_found, iou_threshold=0.2):
  '''objs_found list of list:[
                                [p,c_idx,x,y,w,h],
                                [p,c_idx,x,y,w,h]
                             ]
  '''
  if objs_found.size < 2 or iou_threshold == 1:
    return objs_found

  objs_found = objs_found[np.argsort(
      objs_found[:, 0])[::-1]]  # This was very important

  best_boxes = []
  while len(objs_found) > 0:
    obj = objs_found[0]
    best_boxes.append(list(obj))
    objs_found = objs_found[1:].reshape(-1, 6)

    if len(objs_found) > 0:

      same_class_idxs = np.where(objs_found[:,
                                            1] == obj[1])[0]  # same class_idx
      same_class_objs = objs_found[same_class_idxs].reshape(-1, 6)

      ious = np_iou(obj[None, 2:], same_class_objs[:, 2:])

      delete_idxs = same_class_idxs[np.where(ious >= iou_threshold)[0]]

      objs_found = np.delete(objs_found, delete_idxs, axis=0)

  return best_boxes

