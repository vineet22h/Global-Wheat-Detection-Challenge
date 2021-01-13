## DATA
TRAIN_IMG_PATH          = 'D:\\Machine Learning\\Global Wheat Detection Challenge\\Data\\train_images\\'
TEST_IMG_PATH           = 'D:\\Machine Learning\\Global Wheat Detection Challenge\\Data\\test_images\\'
TRAIN_ANNOT_PATH        = 'D:/Machine Learning/Global Wheat Detection Challenge/Data/clean_GWD .csv'
TRAIN_INSTANCE_PATH     = 'D:/Machine Learning/Global Wheat Detection Challenge/Data/cleaned_train_instances.json'
NUM_CLASSES             = 1



## MODEL
BATCH_SIZE              = 4
NET_H                   = 416
NET_W                   = 416
ANCHOR_PER_SCALE        = 3
STRIDES                 = [8, 16, 32]
MAX_BBOX_PER_SCALE      = 100
FREEZE_DARKNET          = False
IOU_LOSS_THRESH              = 0.7
YOLO_TYPE               = "yolov3"
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                            [[30,  61], [62,   45], [59,  119]],
                            [[116, 90], [156, 198], [373, 326]]]

MASKS                   = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

## LOGS and Checkpoint
CHECKPOINT_PATH = 'model/checkpoint/yolov3_2.h5'
# PRETRAINED_WEIGHTS = './checkpoint/yolov3.weights'

## TEST
OBJECT_THRESH           = 0.40
NMS_THRESH              = 0.7



