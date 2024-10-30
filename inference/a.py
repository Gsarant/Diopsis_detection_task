from ultralytics import YOLOv10
import torch
import supervision as sv
import glob
import shutil
import os, sys
import cv2
from ultralytics import YOLOv10
import torch
import supervision as sv
import numpy as np
import pandas as pd

# from sahi import AutoDetectionModel
# from sahi.utils.cv import read_image
# from sahi.utils.file import download_from_url
# from sahi.predict import get_prediction, get_sliced_prediction, predict

sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import list_boxes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
list_of_names=['Object']

yolo_dicts=[
         {
            'yolo_model_name':'best.pt', 
            'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo', 
            #'methods' :['sahi'], 
            #'methods' :['default','sahi'], 
            'methods' :['default'], 
            'conf':0.3, 
            'image_size':[640,640],
            'remove_metrics_folder':False
        },
        # {
        #     'yolo_model_name':'best.pt', 
        #     'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_synthetic_coco', 
        #     'methods' :['sahi'], 
        #     #'methods' :['default','sahi'], 
        #     #'methods' :['default'], 
        #     'conf':0.3, 
        #     'image_size':[320,320],
        #     'remove_metrics_folder':False
        # },
        # {
        #     'yolo_model_name':'best.pt', 
        #     'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_yolo', 
        #     #'methods' :['default','sahi'], 
        #     'methods' :['default'], 
        #     'conf':0.3, 
        #     'image_size':[640,640],
        #     'remove_metrics_folder':False
        # },
        # {
        #     'yolo_model_name':'best.pt', 
        #     'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_synthetic_yolo', 
        #     #'methods' :['default','sahi'], 
        #     'methods' :['default'], 
        #     'conf':0.3, 
        #     'image_size':[640,640],
        #     'remove_metrics_folder':False
        # },
    ]
results=[]
for yolo_dict in yolo_dicts:
    for method in yolo_dict['methods']:

        # Get subfolders of root folder
        sub_folders = [name for name in os.listdir(yolo_dict['yolo_root_path']) if os.path.isdir(os.path.join(yolo_dict['yolo_root_path'], name))]
        ################################## 
        for sub_folder in sub_folders:
            sub_sub_folders = [name for name in os.listdir(os.path.join(yolo_dict['yolo_root_path'],sub_folder,f"yolov10_run_{sub_folder}"))]
            # Load Yolo Model 
            if method == 'default':
                yolo_model=os.path.join(yolo_dict['yolo_root_path'],sub_folder,f"yolov10_run_{sub_folder}",sub_sub_folders[0],'weights',yolo_dict['yolo_model_name'])
                #yolo_model='../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/yolov10n_b-1_70e_original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/weights/best.pt'
                try:
                    del model
                except:
                    pass
                torch.cuda.empty_cache()
                model = YOLOv10(yolo_model)
                #model.eval()
                
                metrics = model.val(batch=1, conf=0.5, imgsz=640,data='../original_diopsis_data_set_test/data.yml', verbose=False)
                #print("YOLOv8 Validation Results:", yolo_results)
                results_str=f"{sub_folder},{metrics.box.map},{metrics.box.map50},{metrics.box.map75},{metrics.box.maps}"
                results.append(results_str)
for result in results:
    print(result)
