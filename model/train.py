import json
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam

from create_models import create_yolov3, create_yolov4
from losses import yolo_loss
from utils.config import *
from generator import BatchGenerator
from create_train_instance import create_train_instance

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
 
    if not os.path.exists(TRAIN_INSTANCE_PATH):
     create_train_instance(annot_path = TRAIN_ANNOT_PATH,
                           image_path = TRAIN_IMG_PATH,
                           instance_path = TRAIN_INSTANCE_PATH)
 
    train_inst = json.load(open(TRAIN_INSTANCE_PATH, 'r'))

    ##split
    train_inst, test_inst = train_test_split(train_inst, random_state = 0, test_size = 0.1)
    train_inst, valid_inst = train_test_split(train_inst, random_state = 0, test_size = 0.1)
    
    ##Generator
    train_generator = BatchGenerator(train_inst)
    valid_generator = BatchGenerator(valid_inst)
    test_generator = BatchGenerator(test_inst)
    
    ##creating model
    train_model = create_yolov3(input_shape = (NET_H, NET_W, 3), load_pretrained = False)
    train_model.compile(optimizer=Adam(lr=1e-3), 
                        loss= {'yolo_loss': lambda y_true, y_pred: y_pred})
    
    callbacks = [
        ModelCheckpoint(CHECKPOINT_PATH, save_best_only = True, verbose = 1),
        ReduceLROnPlateau(patience = 3, verbose = 1, factor = 0.1)
        # TensorBoard(log_dir = config['train']['tb_path'], write_graph = True, write_images = True)
        ]
    
    ##Fit model
    train_model.fit(x = train_generator,
                    steps_per_epoch = len(train_inst)//BATCH_SIZE,
                    verbose = 1,
                    epochs = 100,
                    validation_data = valid_generator,
                    validation_steps = len(valid_inst)//BATCH_SIZE,
                    callbacks = callbacks)
    
if __name__ == "__main__":
    main()