import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
import numpy as np

from models.yolov4 import YOLOv4
from models.yolov3 import YOLOv3
from losses import decode, yolo_loss
from utils.config import *

def create_yolov4(input_shape, load_pretrained = False, training = True):
    input_layer  = Input(input_shape)
    conv_tensors = YOLOv4(input_layer, NUM_CLASSES)

    model_body = Model(input_layer, conv_tensors)
    model_output = model_body.output
    if load_pretrained:
        model_body.load_weights(CHECKPOINT_PATH, by_name = True, skip_mismatch = True)
        print('Loading weights from {}'.format(CHECKPOINT_PATH))
        
        if FREEZE_DARKNET:
            num = 394
            for i in range(num):
                model_body.layers[i].trainable = False
        
            print('Freeze first {} layers ie Darknet 53 layers'.format(num))
    
    y_true = [
        Input(name='inputt_2', shape=(None, None, 3, (NUM_CLASSES + 5))),  # label_sbbox
        Input(name='inputt_3', shape=(None, None, 3, (NUM_CLASSES + 5))),  # label_mbbox
        Input(name='inputt_4', shape=(None, None, 3, (NUM_CLASSES + 5))),  # label_lbbox
        Input(name='inputt_5', shape=(MAX_BBOX_PER_SCALE, 4)),             # true_sbboxes
        Input(name='inputt_6', shape=(MAX_BBOX_PER_SCALE, 4)),             # true_mbboxes
        Input(name='inputt_7', shape=(MAX_BBOX_PER_SCALE, 4))              # true_lbboxes
    ]
    loss_list = Lambda(yolo_loss, name='yolo_loss',
                        arguments={'num_classes': NUM_CLASSES, 'iou_loss_thresh': 0.7,
                                   'anchors': YOLO_ANCHORS})([*model_output, *y_true])

    model = Model([model_body.input, *y_true], loss_list)
    
    return model

def create_yolov3(input_shape, load_pretrained = False, training = True):
    input_layer  = Input(input_shape)
    conv_tensors = YOLOv3(input_layer, NUM_CLASSES)
    
    model_body = Model(input_layer, conv_tensors)
    model_output = model_body.output
    if load_pretrained:
        model_body.load_weights(CHECKPOINT_PATH, by_name = True, skip_mismatch = True)
        print('Loading weights from {}'.format(CHECKPOINT_PATH))
        
        if FREEZE_DARKNET:
            num = 249
            for i in range(num):
                model_body.layers[i].trainable = False
        
            print('Freeze first {} layers ie Darknet 53 layers'.format(num))
    
    y_true = [
        Input(name='inputt_2', shape=(None, None, 3, (NUM_CLASSES + 5))),  # label_sbbox
        Input(name='inputt_3', shape=(None, None, 3, (NUM_CLASSES + 5))),  # label_mbbox
        Input(name='inputt_4', shape=(None, None, 3, (NUM_CLASSES + 5))),  # label_lbbox
        Input(name='inputt_5', shape=(MAX_BBOX_PER_SCALE, 4)),             # true_sbboxes
        Input(name='inputt_6', shape=(MAX_BBOX_PER_SCALE, 4)),             # true_mbboxes
        Input(name='inputt_7', shape=(MAX_BBOX_PER_SCALE, 4))              # true_lbboxes
    ]
    loss_list = Lambda(yolo_loss, name='yolo_loss',
                        arguments={'num_classes': NUM_CLASSES, 'iou_loss_thresh': 0.7,
                                   'anchors': YOLO_ANCHORS})([*model_output, *y_true])

    model = Model([model_body.input, *y_true], loss_list)
    
    return model
                