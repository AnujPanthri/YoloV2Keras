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