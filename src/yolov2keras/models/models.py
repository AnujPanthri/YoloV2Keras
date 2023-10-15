from .yolov2 import getYoloV2
from .mobilenet import getMobileNet


def get_model(basemodel:str="mobilenet", pretrained:bool=True):
    """
    Return a yolo model with the specified basemodel
    """
    basemodel=basemodel.lower()
    if basemodel == "mobilenet":
        model = getMobileNet(pretrained=pretrained)
        model.basename="mobilenet"
    elif basemodel == "yolov2":
        model = getYoloV2(pretrained=pretrained)
        model.basename="yolov2"
    else:
        model = None
    return model