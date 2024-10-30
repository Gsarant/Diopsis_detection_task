import numpy as np
import cv2 as cv
import json
import os

def get_list_box_yolo_cxcywh(annotation_file:str,image_file:str) -> list:
    list_box=[]
    with open(annotation_file) as f:
        for line in f:
            box = line.strip().split(' ')
            object=box[0]
            h = float(box[4])
            w = float(box[3])
            cx = float(box[1])
            cy = float(box[2])            
            list_box.append({'cx':cx, 'cy':cy, 'w':w, 'h':h, 'object':object})
    return {'image':os.path.basename(image_file) , 'boxes':list_box} 

def get_list_box_yolo(annotation_file:str,image_file:str, img_w:int=None, img_h:int=None) -> list:
    if img_w is None or img_h is None: 
        image=cv.imread(image_file)
        img_w, img_h = image.shape[1],image.shape[0]
    list_box=[]
    with open(annotation_file) as f:
        for line in f:
            box = line.strip().split(' ')
            object=box[0]
            h = float(box[4])
            w = float(box[3])
            cx = float(box[1])
            cy = float(box[2])            
            x1 = float((cx - w/2) * img_w)
            y1 = float((cy - h/2) * img_h)
            x2 = float((cx + w/2) * img_w)
            y2 = float((cy + h/2) * img_h)
            list_box.append({'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'object':object})
    return {'image':os.path.basename(image_file) , 'boxes':list_box} 

def get_list_box_diopsis(annotation_file:str,image_file:str) -> list: 
    list_box=[]
    annotation_json=json.load(open(annotation_file))
    for annotation in annotation_json['annotations']:
        x = annotation['shape']['x']
        y = annotation['shape']['y']
        w = annotation['shape']['width']
        h = annotation['shape']['height']
        object_name=annotation['labels'][0]['name']
        probability = annotation['labels'][0]['probability']
        list_box.append({'x1':x, 'y1':y, 'x2':x+w, 'y2':y+h, 'object':object_name, 'probability':probability})
    return  {'image':os.path.basename(image_file) , 'boxes': list_box}

                    
def get_list_box_coco(annotation_file:str) -> list:
    list_image =  []    
    annotation_json=json.load(open(annotation_file))
    for image in annotation_json['images'][:10]:
        annotation_list_of_image=list(filter(lambda x: x['image_id'] == image['id'], annotation_json['annotations']))
        list_box=[]
        for  annotation  in annotation_list_of_image:
            x = annotation['bbox'][0]
            y = annotation['bbox'][1]
            w = annotation['bbox'][2]
            h = annotation['bbox'][3]
            list_box.append({'x1':x, 'y1':y, 'x2':x+w, 'y2':y+h, 'object':annotation['category_id']})
        list_image.append({'image':os.path.basename(image['file_name']) , 'boxes':list_box})
    return list_image