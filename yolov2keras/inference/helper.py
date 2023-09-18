import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from yolov2keras import config
from .inference import SquarePad

# square_maker=SquarePad(color=(255,255,255))
square_maker = SquarePad(color=(0, 0, 0))


def get_crops(img, objs_found, aligner=None, resize: tuple = None):
    img_h, img_w, _ = img.shape
    all_crops = []
    for obj_found in objs_found:
        xmin, ymin = obj_found[2], obj_found[3]
        xmax, ymax = xmin + obj_found[4], ymin + obj_found[5]
        # rescale them
        xmin, ymin = int(xmin * img_w), int(ymin * img_h)
        xmax, ymax = int(xmax * img_w), int(ymax * img_h)

        crop = img[ymin:ymax, xmin:xmax]
        if aligner is not None:
            crop = aligner.align((crop,))[0]
            if crop is None:
                continue
        if resize is not None:
            crop = square_maker(crop)
            crop = cv2.resize(crop, resize)
        all_crops.append(crop)

    return all_crops


def rescale(obj_found, w, h):
        # xywh
        obj_found[0] *= w
        obj_found[1] *= h
        obj_found[2] *= w
        obj_found[3] *= h
        return obj_found


def show_objects(img, objs_found):
    if (type(img) == str):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    
    plt.figure()
    plt.imshow(img)
    for i in range(len(objs_found)):
        # p,c_idx,x,y,w,h
        p = objs_found[i][0]
        obj_name = objs_found[i][1]
        obj = rescale(objs_found[i][2:], img.shape[1], img.shape[0])
        
        plt.gca().add_patch(
            Rectangle((obj[0], obj[1]), (obj[2]), (obj[3]),
                      linewidth=4,
                      edgecolor=config.class_colors[obj_name],
                      facecolor='none'))
        plt.text(obj[0], obj[1], obj_name)
    plt.show()


def pred_image(img, objs_found, font_scale=2, thickness=4):
    if (type(img) == str):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    
    
    for i in range(len(objs_found)):
        # p,c_idx,x,y,w,h
        p = objs_found[i][0]
        obj_name = objs_found[i][1]
        obj = rescale(objs_found[i][2:], img.shape[1], img.shape[0])

        img = cv2.rectangle(img, (int(obj[0]), int(obj[1])),
                            (int(obj[0] + obj[2]), int(obj[1] + obj[3])),
                            (config.class_colors[obj_name] * 255), thickness)
        img = cv2.putText(img,
                          obj_name, (int(obj[0]), int(obj[1])),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale, (0, 0, 0),
                          thickness,
                          lineType=cv2.LINE_AA)
        # draw_text(img, "world", font_scale=4, pos=(10, 20 + h), text_color_bg=(255, 0, 0))
    return img
