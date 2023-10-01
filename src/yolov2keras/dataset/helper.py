import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
from yolov2keras import config



def show_examples(dataset,num_examples):
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
      obj_name=config.idx_to_class[objs_names[obj_idx]]
      plt.gca().add_patch(Rectangle(( center_x-(width/2),center_y-(height/2) ),width,height,linewidth=2,edgecolor=config.class_colors[obj_name],facecolor='none'))
      plt.text( center_x-(width/2) , center_y-(height/2) ,obj_name)
    
  rows=math.ceil(num_examples/5)
  cols=5
  
  fig = plt.figure(figsize=(cols*5,rows*5))
  for i,data in enumerate(dataset.take(num_examples)):
      fig.add_subplot(rows,cols,i+1)
      display_one_image(data)
  plt.show()


def show_anchors(anchor_boxes):

  fig=plt.figure(figsize=(5*config.num_anchors,5*1))
  for i,anchor_box in enumerate(anchor_boxes):
    fig.add_subplot(1,config.num_anchors,i+1)
    plt.imshow(np.zeros((int(config.output_size),int(config.output_size),3)))
    # plt.imshow(np.zeros((4,4,3)))
    plt.gca().add_patch(Rectangle((1,1),(anchor_box[0]),(anchor_box[1]),linewidth=4,edgecolor=np.random.rand(3),facecolor='none'))
    plt.text(1,1,f'w:{np.round(anchor_box[0],2)},h:{np.round(anchor_box[1],2)}',bbox=dict(edgecolor='none',facecolor='white', alpha=1))
  plt.show()


def show_yolo_examples(dataset):
  
  def show_images(data):
    plt.imshow(data[0].numpy().astype('uint8'))
    idxs=np.where(data[1].numpy()[...,0]==1)
    if np.size(idxs):
      for i,obj in enumerate(data[1].numpy()[idxs[0],idxs[1],idxs[2],:]):
        obj=obj[1:]
        obj[4]=np.argmax(obj[4:])
        obj=obj[:5]
        obj[:-1]*=config.cell_size # scaling back w and h


        obj[1]=(idxs[0][i]*config.cell_size)+obj[1]  # center y
        obj[0]=(idxs[1][i]*config.cell_size)+obj[0]  # center x


        obj[0]=obj[0]-(obj[2]/2)  # xmin
        obj[1]=obj[1]-(obj[3]/2)  # ymin

        obj_name=config.idx_to_class[obj[4]]
        plt.gca().add_patch(Rectangle((obj[0],obj[1]),(obj[2]),(obj[3]),linewidth=1,edgecolor=config.class_colors[obj_name],facecolor='none'))
        plt.text(obj[0],obj[1],obj_name)


  rows=math.ceil(dataset.element_spec[1].shape[0]/5)
  cols=5
  fig = plt.figure(figsize=(cols*5,rows*5))
  
  for batchdata in dataset.take(1):
    batchdata
  for i in range(batchdata[0].shape[0]):
    fig.add_subplot(rows,cols,i+1)
    show_images((batchdata[0][i],batchdata[1][i]))
  plt.show()