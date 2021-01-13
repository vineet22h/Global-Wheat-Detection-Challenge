from albumentations import *
import cv2
from tensorflow.keras.utils import Sequence
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.image import convert_image_dtype
import tensorflow as tf

from utils.config import *
from losses import bbox_iou

class BatchGenerator(Sequence):
    def __init__(self, train_inst, shuffle = True):
        self.train_inst = train_inst
        self.batch_size = BATCH_SIZE
        self.num_classes = NUM_CLASSES
        self.shuffle = shuffle      
        self.input_net_h = NET_H
        self.input_net_w = NET_W
        self.output_net_h = [self.input_net_h//s for s in STRIDES]
        self.output_net_w = [self.input_net_w//s for s in STRIDES]
        self.anchor_per_scale = ANCHOR_PER_SCALE
        self.max_bbox_per_scale = MAX_BBOX_PER_SCALE
        self.anchors = YOLO_ANCHORS
        self.stride = np.array(STRIDES)
        if self.shuffle :
            np.random.shuffle(self.train_inst)
    
    def load_img(self, path):
        return cv2.imread(path)
    
    def __len__(self):
        return math.ceil(len(self.train_inst) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle :
            np.random.shuffle(self.train_inst)
            
    def apply_aug(self, img, bboxes):
        labels = np.ones((len(bboxes), ))
        aug = Compose([
            OneOf([
                RandomSizedCrop(
                    min_max_height=(200, 200), 
                    height=self.input_net_h, 
                    width=self.input_net_w, 
                    p=0.5
                    ),
                Resize(self.input_net_h, self.input_net_w, p=0.5),
            ], p=1),
            
            OneOf([
                Flip(),
                RandomRotate90(),
            ], p=1),
            
            OneOf([
                HueSaturationValue(),
                RandomBrightnessContrast()
            ], p=1),
            OneOf([
                GaussNoise(),
                GlassBlur(),
                ISONoise(),
                MultiplicativeNoise(),
            ], p=0.5),
            Cutout(
                num_holes=8, 
                max_h_size=16, 
                max_w_size=16, 
                fill_value=0, 
                p=0.5
            ), 
            CLAHE(p=0.5),
            ToGray(p=0.5),
            
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        
        aug_result = aug(image=img, bboxes=bboxes, labels = labels)
    
        return aug_result['image'], aug_result['bboxes']
    
    
    def preprocess_bbox(self, bboxes):
        label = [np.zeros((self.output_net_h[i], self.output_net_h[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        
        for bbox in bboxes:
            bbox = np.array(bbox)
            bbox_coor = bbox[:4]
            bbox_class_ind = 0

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            ## pascal voc to yolo
            bbox_xywh = np.concatenate([(bbox_coor[:2] + bbox_coor[2:])*0.5, (bbox_coor[2:] - bbox_coor[:2])], axis=-1)
            
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.stride[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True
                    
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                
                
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
    
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
            
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __getitem__(self, idx):     
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size
        
        if r_bound > len(self.train_inst):
            r_bound = len(self.train_inst)
            l_bound = r_bound - len(self.train_inst)
        
        batch_image = np.zeros((self.batch_size, self.input_net_h, self.input_net_w, 3), dtype=np.uint8)

        batch_label_sbbox = np.zeros((self.batch_size, self.output_net_h[0], self.output_net_w[0],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float16)
        
        batch_label_mbbox = np.zeros((self.batch_size, self.output_net_h[1], self.output_net_w[1],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float16)
        
        batch_label_lbbox = np.zeros((self.batch_size, self.output_net_h[2], self.output_net_w[2],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float16)

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float16)
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float16)
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float16)


        for ind, i in enumerate(range(l_bound, r_bound)):   
            if ind >= self.batch_size or ind < 0:
                continue
            
            img = self.load_img(self.train_inst[i]['filename'])
            bbox = self.train_inst[i]['BBox']
            img, bbox = self.apply_aug(img, bbox)
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_bbox(bbox)
            batch_image[ind, :, :, :] = img
            batch_label_sbbox[ind, :, :, :, :] = label_sbbox
            batch_label_mbbox[ind, :, :, :, :] = label_mbbox
            batch_label_lbbox[ind, :, :, :, :] = label_lbbox
            batch_sbboxes[ind, :, :] = sbboxes
            batch_mbboxes[ind, :, :] = mbboxes
            batch_lbboxes[ind, :, :] = lbboxes
     
        
        return [batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes], np.zeros(self.batch_size)



    
        
        