import tensorflow as tf
from model.utils.config import *
from model.models.yolov3 import YOLOv3
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import cv2
import numpy as np
from model.utils.helper import sigmoid, show_img, resize
# from model.utils.evaluate_map import get_map

def scale_to_bbox(scale, anchors, mask):
    grid_h, grid_w, num_boxes = map(int, scale.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)
    
    # Reshape to batch, height, width, num_anchors, box_params.
    scale = scale[0]
    box_xy = sigmoid(scale[..., :2])
    box_wh = np.exp(scale[..., 2:4])
    box_wh = box_wh * anchors_tensor
    
    box_confidence = sigmoid(scale[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(scale[..., 5:])
    
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    
    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (NET_H, NET_W)
    box_xy -= (box_wh / 2.)   
    boxes = np.concatenate((box_xy, box_wh), axis=-1)
    
    box_scores = box_confidence * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJECT_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores
    
def post_process_bbox(scale_1, scale_2, scale_3):
    boxes, classes, scores = [], [], []
    for scale, mask in zip([scale_1, scale_2, scale_3], MASKS):
        b, c, s = scale_to_bbox(scale, np.array(YOLO_ANCHORS).reshape(-1, 2), mask)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
    
    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
    
    # Scale boxes back to original image shape.
    w, h = NET_W, NET_H
    image_dims = [w, h, w, h]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

    return boxes, scores, classes

def nms( boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep

def predict(model, img):
    print('in predict')
    img = cv2.resize(img, (416, 416))
    print('resized image',img)
    pred = model.predict(tf.expand_dims(img, axis = 0))
    print('predicted')
    output_net_h = [NET_H//s for s in STRIDES]
    output_net_w = [NET_W//s for s in STRIDES]

    
    scale_1 = np.reshape(pred[2], (1, output_net_h[0], output_net_w[0],
                                   3, 5+NUM_CLASSES))
    scale_2 = np.reshape(pred[1], (1, output_net_h[1], output_net_w[1],
                                   3, 5+NUM_CLASSES))
    scale_3 = np.reshape(pred[0], (1, output_net_h[2], output_net_w[2],
                                   3, 5+NUM_CLASSES))
    
    bboxes, scores, classes = post_process_bbox(scale_1, scale_2, scale_3)
    
    return bboxes, scores, classes

# if ___name__ == '__main__':
#     input_layer  = Input((NET_H, NET_W, 3))
#     output_layer = YOLOv3(input_layer, NUM_CLASSES)
#     model = Model(input_layer, output_layer)
    
#     model.load_weights(CHECKPOINT_PATH)
#     img = cv2.resize(cv2.imread(train_inst[5]['filename']), (416, 416))
#     bboxes, scores, cl = predict(model, img)
#     show_img(img, bboxes, scores)

