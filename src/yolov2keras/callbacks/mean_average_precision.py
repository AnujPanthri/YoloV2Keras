import yolov2keras as yod
from yolov2keras import config
import numpy as np
import tensorflow as tf

def get_all_data(model, ds, p_thres=0.5):
    idx = 0
    y_true = []
    y_pred = []

    get_true_objects = lambda x, example_idx: yod.inference.get_objects(
        x, decode_preds=False, example_idx=example_idx
    )
    get_pred_objects = lambda x, example_idx: yod.inference.get_objects(
        x, p_thres, decode_preds=True, example_idx=example_idx
    )

    for data in ds:
        # print(data[0].shape)
        # print(data[1].shape)
        y_preds_raw = model.predict(data[0], verbose=0)
        batch_size = data[1].shape[0]
        for i in range(batch_size):
            y_true.extend(get_true_objects(data[1][i].numpy(), idx))
            y_pred.extend(get_pred_objects(y_preds_raw[i], idx))

            idx += 1

    return y_true, y_pred


def calculate_MAP_for_class(y_true, y_pred, class_idx,iou_thres=0.5):
    """
    calculates the Mean Average Precision for a particular class(category)

    y_true: [ [train_idx,conf,class_idx,xmin,ymin,w,h] , ... ]
    y_pred: [ [train_idx,conf,class_idx,xmin,ymin,w,h] , ... ]

    returns the recall and precision lists


    """
    epsilon = 1e-6

    detections = []
    ground_truths = []

    #   filter detections and ground_truths boxes only coresponding to a particular class

    for ground_truth in y_true:
        if ground_truth[2]==class_idx:
            ground_truths.append(ground_truth) 

    for detection in y_pred:
        if detection[2]==class_idx:
            detections.append(detection)

    # calculate the number of ground truth boxes for each train_idx

    amount_boxes = dict()
    for ground_truth in ground_truths:
        if ground_truth[0] not in amount_boxes:
            amount_boxes[ground_truth[0]]=1
        else:
            amount_boxes[ground_truth[0]]+=1

    # now zero hot encode the the number of ground truth boxes per train_idx

    for key,val in amount_boxes.items():
        amount_boxes[key]=np.zeros(val)

    # now sort the detections on the basis of conf in decreasing order 
    detections.sort(key=lambda x:x[1],reverse=True)

    TP=np.zeros(len(detections))
    FP=np.zeros(len(detections))
    total_true_boxes = len(ground_truths)

    if total_true_boxes==0: return None,None

    for detection_idx,detection in enumerate(detections):
        # find if this detection corresponds to FP or TP

        ground_truth_img=[gt for gt in ground_truths if gt[0]==detection[0]]

        if len(ground_truth_img)==0:
            FP[detection_idx]=1
            continue
        
        ious = yod.inference.np_iou(np.array(detection[3:])[None,:],np.array(ground_truth_img)[:,3:])
        
        # print(len(ground_truth_img))
        # print(ious)
        best_gt_idx=np.argmax(ious)
        best_iou=ious[best_gt_idx]
        # print(best_gt_idx,best_iou)
        if best_iou>=iou_thres:
            if amount_boxes[detection[0]][best_gt_idx]==0:
                TP[detection_idx]=1
                amount_boxes[detection[0]][best_gt_idx]=1
            else:
                FP[detection_idx]=1

        else:
            FP[detection_idx]=1
        # return 1

    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(FP)

    recalls = TP_cumsum / (total_true_boxes+epsilon)
    precisions = TP_cumsum / (TP_cumsum+FP_cumsum+epsilon)
    
    recalls = np.r_[np.array(0),recalls] # x axis
    precisions = np.r_[np.array(1),precisions] # y axis

    return recalls,precisions

def calculate_MAP(y_true, y_pred,iou_thres=0.5):

    APs=[]

    for class_idx in range(len(config.classnames)):
                
        recalls,precisions=yod.callbacks.calculate_MAP_for_class(y_true,y_pred,class_idx=class_idx,iou_thres=iou_thres)
        if recalls is None and precisions is None:
            continue
        APs.append(np.trapz(precisions,recalls))

    return APs

class MAPCallback(tf.keras.callbacks.Callback):
    def __init__(self,ds,iou_thres=0.5,per_nth_epoch=5):
        self.ds=ds
        self.best_map=0.0
        self.iou_thres=iou_thres
        self.per_nth_epoch=per_nth_epoch

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.per_nth_epoch==0:
            y_true,y_pred=yod.callbacks.get_all_data(self.model,self.ds,p_thres=0.01)
            APs=calculate_MAP(y_true,y_pred,iou_thres=self.iou_thres)
            mAP=np.mean(APs)            
            tf.print(f"\nmap@iou{self.iou_thres}:",mAP)

        # return super().on_epoch_end(epoch, logs)
