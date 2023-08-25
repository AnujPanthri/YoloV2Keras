import struct
import numpy as np
import YoloObjectDetection as yod
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


# tf.keras.saving.get_custom_objects().clear()
@tf.keras.utils.register_keras_serializable("yolov2") # 1
class yolo_reshape(tf.keras.layers.Layer):

  def __init__(self,num_anchors,last_item, **kwargs):
    super(yolo_reshape, self).__init__(**kwargs)
    self.last_item=last_item
    self.num_anchors=num_anchors

  def call(self,output_layer):
    shape = [tf.shape(output_layer)[k] for k in range(4)]
    return tf.reshape(output_layer,[shape[0],shape[1],shape[2],self.num_anchors,self.last_item])

  def compute_output_shape(self, input_shape):
    return (input_shape[0],input_shape[1],input_shape[2],self.num_anchors,self.last_item)


  def get_config(self):
    config = super(yolo_reshape, self).get_config()
    config.update(
        {
            "last_item": self.last_item,
            "num_anchors": self.num_anchors
        }
    )
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  




def space_to_depth_x2(x):
  return tf.nn.space_to_depth(x,block_size=2)


def getYolov2(pretrained=True):
    
    x_input=layers.Input(shape=(None,None,3))
    # x_input=layers.Input(shape=(608,608,3))
    x=layers.Lambda(lambda x:x/255.)(x_input)
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',name='conv_1',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_1')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
    x=layers.MaxPooling2D(pool_size=(2,2))(x)

    x=layers.Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv_2',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_2')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
    x=layers.MaxPooling2D(pool_size=(2,2))(x)

    x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv_3',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_3')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv_4',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_4')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv_5',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_5')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
    x=layers.MaxPooling2D(pool_size=(2,2))(x)

    x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv_6',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_6')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(128,(1,1),strides=(1,1),padding='same',name='conv_7',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_7')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv_8',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_8')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)
    x=layers.MaxPooling2D(pool_size=(2,2))(x)

    x=layers.Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_9',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_9')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv_10',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_10')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_11',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_11')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv_12',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_12')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv_13',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_13')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x=layers.MaxPooling2D(pool_size=(2,2))(x)

    x=layers.Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_14',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_14')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv_15',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_15')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_16',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_16')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv_17',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_17')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_18',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_18')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_19',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_19')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_20',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_20')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    skip_connection=layers.Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv_21',use_bias=False)(skip_connection)
    skip_connection=layers.BatchNormalization(name='norm_21')(skip_connection)
    skip_connection=layers.LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection=layers.Lambda(space_to_depth_x2)(skip_connection) # halfs the resolution and add more depth

    x=layers.concatenate([skip_connection,x])
    x=layers.Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv_22',use_bias=False)(x)
    x=layers.BatchNormalization(name='norm_22')(x)
    x=layers.LeakyReLU(alpha=0.1)(x)

    x=layers.Conv2D((yod.num_anchors*(5+len(yod.classnames))),(1,1),strides=(1,1),padding='same',name='conv_23')(x)
    out=yolo_reshape(yod.num_anchors,(5+len(yod.classnames)))(x)

    model = Model(x_input,out,name='yolo_v2_model')
    
    
    path_to_weight = get_file_path("https://pjreddie.com/media/files/yolov2.weights");offset=4;nb_conv = 23;  # (trained on coco dataset (80 classes))
    # path_to_weight = "./yolov2-voc.weights";offset=5;nb_conv = 23;
    # path_to_weight = 'darknet19_448.conv.23';offset=4;nb_conv = 18;
    model = load_yolo_weights(model,path_to_weight,offset,nb_conv)
    return model


def get_file_path(url):
   # returns file path 
   # downloads file from the given url if not already present
   # should work on both os
   pass
   


def load_yolo_weights(model,path_to_weight,offset,nb_conv):
    class WeightReader:
        def __init__(self, weight_file):
            self.offset = offset # an offset of 5 as first 5 values are non weight values(they are weight header)(for yolov2-voc.weights)
    #         self.all_weights = np.fromfile(weight_file, dtype='float32')
            self.all_weights = open(weight_file,'rb')
            print("weight Header(major, minor, revision, seen):",struct.unpack(f'{offset}i', self.all_weights.read(offset*4)))

        def read_bytes(self, size):
            weights = struct.unpack('%df' % size, self.all_weights.read(size*4))
    #         print(weights)
    #         input("wait now forever")
            return np.array(weights)

    weight_reader = WeightReader(path_to_weight)
    print("all_weights = {}".format(np.fromfile(path_to_weight, dtype='float32').shape[0]-weight_reader.offset))

    for i in range(1, nb_conv+1):
        conv_layer = model.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            print(kernel.shape)
            kernel=kernel.reshape([*kernel.shape[:-1],yod.num_anchors,-1]) # reshape to this format so we change change position of p idx
            idx=4 # in darknet each object was encoded as [x,y,w,h,p,c] but we use [p,x,y,w,h,c]
            kernel=np.concatenate([kernel[...,idx:idx+1],kernel[...,:idx],kernel[...,idx+1:]],axis=-1)  # setting p to idx 0
            print(kernel.shape)
            kernel=kernel.reshape([*kernel.shape[:-2],-1])
            print(kernel.shape)
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])
    

    layer   = model.layers[-2] # the last convolutional layer
    weights = layer.get_weights()
    print(layer.name)
    new_kernel = np.random.normal(size=weights[0].shape)/(yod.output_size*yod.output_size)
    new_bias   = np.random.normal(size=weights[1].shape)/(yod.output_size*yod.output_size)

    layer.set_weights([new_kernel, new_bias])

    return model