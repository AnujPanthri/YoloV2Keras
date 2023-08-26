import yolov2keras as yod
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

def find_anchors(dataset):
    
    all_width_height=np.array([])
    for data in tqdm(dataset):
        image_height,image_width=data[0].numpy().shape[:2]
        boxes=data[2].numpy()  # center_x,center_y,width,height
        widths,heights = boxes[:,2]*image_width , boxes[:,3]*image_height
        if all_width_height.shape[0]==0:
            all_width_height=np.c_[widths,heights]
        else:
            all_width_height=np.r_[all_width_height,np.c_[widths,heights]]
    # all_width_height.shape
    
    
    w_h=all_width_height
    kmeans=KMeans(yod.num_anchors).fit(w_h)
    z=kmeans.predict(w_h)
    

    anchor_boxes = kmeans.cluster_centers_

    scale = yod.output_size / yod.input_size
    anchor_boxes=anchor_boxes*scale  #  (anchor_boxes/image_size)*output_size

    return anchor_boxes