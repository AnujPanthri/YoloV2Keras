import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
import YoloObjectDetection as yod


def display_one_image(data):
  image,objs_names,objs=data
  image=image.numpy()
  objs_names=objs_names.numpy()
  objs=objs.numpy()
  plt.imshow(image.astype('uint8'))
  image_height,image_width=image.shape[:2]
  for obj_idx in range(len(objs)):
    center_x,center_y,width,height=objs[obj_idx]
    center_x,center_y,width,height = center_x*image_width , center_y*image_height , width*image_width , height*image_height
    obj_name=yod.idx_to_class[objs_names[obj_idx]]
    plt.gca().add_patch(Rectangle(( center_x-(width/2),center_y-(height/2) ),width,height,linewidth=2,edgecolor=yod.class_colors[obj_name],facecolor='none'))
    plt.text( center_x-(width/2) , center_y-(height/2) ,obj_name)


def show_examples(dataset,num_examples):
    
    rows=math.ceil(num_examples/5)
    cols=5
    
    fig = plt.figure(figsize=(cols*5,rows*5))
    for i,data in enumerate(dataset.take(num_examples)):
        fig.add_subplot(rows,cols,i+1)
        display_one_image(data)
    plt.show()


def show_anchors(anchor_boxes):

  fig=plt.figure(figsize=(5*yod.num_anchors,5*1))
  for i,anchor_box in enumerate(anchor_boxes):
    fig.add_subplot(1,yod.num_anchors,i+1)
    plt.imshow(np.zeros((int(yod.output_size),int(yod.output_size),3)))
    # plt.imshow(np.zeros((4,4,3)))
    plt.gca().add_patch(Rectangle((1,1),(anchor_box[0]),(anchor_box[1]),linewidth=4,edgecolor=np.random.rand(3),facecolor='none'))
    plt.text(1,1,f'w:{np.round(anchor_box[0],2)},h:{np.round(anchor_box[1],2)}',bbox=dict(edgecolor='none',facecolor='white', alpha=1))
  plt.show()