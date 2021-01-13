import pandas as pd
import json
from tqdm import tqdm

def create_train_instance(annot_path, image_path, instance_path):
    print("creating Training instances")
    df = pd.read_csv(annot_path)
    all_imgs = []
    img = {'filename':'',
           'height':0,
           'width':0, 
           'BBox':[],
           'class': []
           }
    
    for i in tqdm(range(1,df.shape[0])):
        
        filename = df.loc[i-1]['image_id']
        img['filename'] = image_path+filename+'.jpg'
        img['height'] = int(df.loc[i-1]['height'])
        img['width'] = int(df.loc[i-1]['width'])
        img['class'] = []
        
        if df.loc[i-1]['image_id'] == df.loc[i]['image_id']:
            box, class_name = df.loc[i-1]['bbox'], df.loc[i-1]['source']
            
            box = [i[1:] for i in box.split(',')]
            box[-1] = box[-1][0:-1]
            box = [float(i) for i in box]
            box[2] += box[0]
            box[3] += box[1]
            
            
            img['BBox'] +=[box]
            img['class'] +=[0]
            
        else:
            box, class_name = df.loc[i-1]['bbox'], df.loc[i-1]['source']
            
            box = [i[1:] for i in box.split(',')]
            box[-1] = box[-1][0:-1]
            box = [float(i) for i in box]
            box[2] += box[0]
            box[3] += box[1]
            
            img['BBox'] +=[box]
            img['class'] +=[0]
            all_imgs.append(img)
            img = {'filename':'',
                   'height':0,
                   'width':0, 
                   'BBox':[]
                   }
            
            
        if i == df.shape[0]-1:
            all_imgs+=[img]
    with open(instance_path, "w") as outfile:  
      json.dump(all_imgs, outfile, indent = 1)    
