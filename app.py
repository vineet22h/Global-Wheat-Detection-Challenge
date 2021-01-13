from flask import Flask, render_template, flash, redirect, url_for, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
from model.test import predict, post_process_bbox
from model.utils.helper import show_img
from model.utils.config import *
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from model.models.yolov3 import YOLOv3

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

input_layer  = Input((NET_H, NET_W, 3))
output_layer = YOLOv3(input_layer, NUM_CLASSES)
model = Model(input_layer, output_layer)

model.load_weights(CHECKPOINT_PATH)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
all_imgs = []
curr_index = 0
leaf_count = {}

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/')
def upload_form():
    print('in upload_form')
    return render_template('main.html')

@app.route('/file', methods=['POST'])
def upload_image():
    print('in upload_image')
    global curr_index
    imgs = request.files
    if len(imgs) == 0:
        response = jsonify({'response': False})
        return response
    print('images', imgs)
    for img in imgs:
        file = imgs[img]
        filename = file.filename
        print(filename)
        filename = secure_filename(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        all_imgs.append(filename)
    detect()
    curr_image = all_imgs[curr_index]

    return jsonify({'filename': curr_image,
                    'pred':'static/preditions/'+curr_image})

def detect():
    global leaf_count
    ##pred
    
    curr_image = all_imgs[curr_index]
    img = plt.imread('static/uploads/'+curr_image)
    print(img)
    
    img = cv2.resize(img, (416, 416))
    print('resized image',img)
    print(model)
    pred = model.predict(tf.expand_dims(img, axis = 0))
    print('predicted', pred)
    output_net_h = [NET_H//s for s in STRIDES]
    output_net_w = [NET_W//s for s in STRIDES]

    scale_1 = np.reshape(pred[2], (1, output_net_h[0], output_net_w[0],
                                   3, 5+NUM_CLASSES))
    scale_2 = np.reshape(pred[1], (1, output_net_h[1], output_net_w[1],
                                   3, 5+NUM_CLASSES))
    scale_3 = np.reshape(pred[0], (1, output_net_h[2], output_net_w[2],
                                   3, 5+NUM_CLASSES))
    
    bboxes, scores, classes = post_process_bbox(scale_1, scale_2, scale_3)
    print('show_img')
    leaf_count[curr_image] = len(bboxes)
    show_img(img, bboxes, scores, save = True, img_name = 'static/predictions/'+curr_image)
    a = plt.imread('static/predictions/'+curr_image)

    print('save', a.shape)
    plt.imsave('static/predictions/'+curr_image, a[600:4450,640:4490])


@app.route('/next')
def next():
    global curr_index 
    print('in next')
    curr_index+=1
    if curr_index == len(all_imgs)-1:
        detect()
        return jsonify({'response':False, 
                        'filename': all_imgs[curr_index]})

    if not os.path.exists(all_imgs[curr_index]):
        detect()
    return jsonify({'response':True, 
                    'filename': all_imgs[curr_index]})

@app.route('/previous')
def previous():
    global curr_index
    print('in previous') 
    curr_index-=1
    print(curr_index)
    if curr_index == 0:
        detect()
        return jsonify({'response':False, 
                        'filename': all_imgs[curr_index]})

    if not os.path.exists(all_imgs[curr_index]):
        detect()
    return jsonify({'response':True, 
                    'filename': all_imgs[curr_index]})

@app.route('/assign_img', methods = ["POST"])
def assign_img():
    print('in assign_img')
    print(request)
    model_name = request.form['model_name']
    print(model_name)
    print(leaf_count)
    if model_name == 'none':
        return jsonify({'filename': '/static/uploads/'+all_imgs[curr_index],
                        'leaf_count': '--'})

    return jsonify({'filename': '/static/predictions/'+all_imgs[curr_index],
                    'leaf_count': leaf_count[all_imgs[curr_index]]})

@app.route('/reload')
def reload():
    global all_imgs
    global curr_index
    print("in reload")
    print(all_imgs)
    for img in all_imgs:
        os.unlink('static/uploads/'+img)
        try:
            os.unlink('static/predictions/'+img)
        except:
            pass
    all_imgs = []
    curr_index = 0
    return jsonify({'response': True})

if __name__ == "__main__":
    app.run(port = 7555)