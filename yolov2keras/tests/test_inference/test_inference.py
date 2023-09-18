# from yolov2keras.utils import iou
from yolov2keras import inference
import pytest
import numpy as np


def test_SquarePad():

    squareobj=inference.SquarePad()
    img = (np.random.rand(500*1000*3).reshape(500,1000,3)*255).astype("uint8")
        
    processed_img=squareobj(img)
    objs_found=np.array([[1,1,0.3,0.3,1,1],    #[P,C_IDX,CENTER_X,CENTER_Y,W,H]
                         
                         ]) 
    
    rescaled_objs_found=squareobj.rescale(objs_found)

    assert processed_img.shape[0]-processed_img.shape[1]<=1  #it is converted to a square
    assert processed_img.shape[0]>=img.shape[0]     # it is only padded if needed
    assert processed_img.shape[1]>=img.shape[1]     # it is only padded if needed
    assert np.all(rescaled_objs_found>=0)    # even after rescaling the coordinates should stay within 0 to 1
    assert np.all(rescaled_objs_found<=1)    # even after rescaling the coordinates should stay within 0 to 1
