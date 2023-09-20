import gradio as gr
import json
from glob import glob
import yolov2keras as yod


model_path="output/v1/"
object_detector = yod.load_model(model_path)
object_detector.set_config(p_thres=0.5,nms_thres=0.3,image_size=[416])

def get_output(img,p_thres,nms_thres,image_size):

    object_detector.set_config(p_thres=p_thres,nms_thres=nms_thres,image_size=[image_size])

    detections = object_detector.predict(img)
    # print(detections)
    pred_img=yod.inference.helper.pred_image(img,detections)

    return pred_img,{"number of objects:":len(detections),'objects_found':detections}



app=gr.Interface(get_output,inputs=[
                                    # gr.Image(),
                                    gr.Image(streaming=True,source='webcam'),
                                    gr.Slider(0,1,value=0.6,label='min_confidence'),
                                    gr.Slider(0,1,value=0.3,label='nms_iou_threshold'),
                                    gr.Slider(256,2080,step=32,value=256,label='image_size'),],

                                    outputs=[gr.Image(label='Objects Found'),gr.Text(label='Objects_found')],
                                    title="Yolo V2 Face detection(sized-mode)",
                                    description=f"we can detection objects which are: {', '.join(yod.config.classnames)}",
                                    # examples=[[item] for item in glob('imgs/*')],
                                    live=True
                                    )
app.launch(debug=True)