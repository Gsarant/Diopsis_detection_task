import glob
import shutil
import os, sys
from ultralytics import YOLOv10
import torch
import pandas as pd
import numpy as np
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import list_boxes
from metrics.object_detection_metrics import Object_detection_metrics
import inf_types

def get_metrics(preds,targets,device):
    mobj=Object_detection_metrics(preds,targets,device=device)
    print('Len Preds :',len(preds[0]['boxes']))
    print('Len Targets :',len(targets[0]['boxes']))
    # if len(preds[0]['boxes']) == len(targets[0]['boxes']):
    #     confusion_matrix=mobj.confusion_matrix()
    #     f1=mobj.F1()
    #     precision=mobj.Precision()
    #     recal=mobj.Recall()
    map_calculate=mobj.map_calculate()
   
    dict_metrics=dict()
    for item in map_calculate:
        try:
            dict_metrics[item.split(':')[0]]=float(item.split(':')[1]).__round__(4)
        except:
            dict_metrics[item.split(':')[0]]=None 

    return metrics,dict_metrics

def get_metrics_str_for_csv(image_file_name,number_of_images,preds,targets,device):
    metrics,dict_metrics= get_metrics(preds,targets,device)
    if number_of_images == 1 and not image_file_name is None: 
        metrcs_str=f"{os.path.basename(image_file_name)},{len(preds[0]['boxes'])},{len(targets[0]['boxes'])}"
    else:
        metrcs_str=f"{number_of_images},{len(preds[0]['boxes'])},{len(targets[0]['boxes'])}"
    for m in metrics:
        metrcs_str=metrcs_str+','+ str(dict_metrics[m])
    print(metrcs_labels_str)
    print(metrcs_str)
    return metrcs_str

#Initialize     
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
list_of_names=['Object']

# Dictionaries
yolo_dicts=[
         {
            'yolo_model_name':'best.pt', 
            'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo', 
            #'methods' :['sahi'], 
            'methods' :['default','sahi','ensemble'], 
            #'methods' :['default'], 
            #'methods' :['ensemble'], 
            'conf':0.25, 
            'image_size':[640,640],
            'remove_metrics_folder':False
        },
        
        # {
        #     'yolo_model_name':'last.pt', 
        #     'yolo_root_path'  :'../kfolds/average_diopsis', 
        #     'methods' :['average'], 
        #     'average_yolo_models':[
        #                             '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/yolov10n_b-1_70e_original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/weights/best.pt',
        #                             '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_1_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_cropted_1_yolo/yolov10n_b-1_70e_original_diopsis_data_set_train_val_synthetic_cropted_1_yolo/weights/best.pt',
        #                             '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_2_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_cropted_2_yolo/yolov10n_b-1_70e_original_diopsis_data_set_train_val_synthetic_cropted_2_yolo/weights/best.pt',
        #                             '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_3_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_cropted_3_yolo/yolov10n_b-1_70e_original_diopsis_data_set_train_val_synthetic_cropted_3_yolo/weights/best.pt',
        #                             '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_4_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_cropted_4_yolo/yolov10n_b-1_70e_original_diopsis_data_set_train_val_synthetic_cropted_4_yolo/weights/best.pt',
                                    
        #                             #'../kfolds/original_diopsis_data_set_train_val_synthetic_coco/original_diopsis_data_set_train_val_synthetic_1/yolov10_run_original_diopsis_data_set_train_val_synthetic_1/yolov10n_b-1_50e_original_diopsis_data_set_train_val_synthetic_1/weights/best.pt',
        #                             #'../kfolds/original_diopsis_data_set_train_val_synthetic_coco/original_diopsis_data_set_train_val_synthetic_2/yolov10_run_original_diopsis_data_set_train_val_synthetic_2/yolov10n_b-1_50e_original_diopsis_data_set_train_val_synthetic_2/weights/best.pt',
        #                             #'../kfolds/original_diopsis_data_set_train_val_synthetic_yolo/original_diopsis_data_set_train_val_synthetic_1_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_1_yolo/yolov10n_b16_70e_original_diopsis_data_set_train_val_synthetic_1_yolo/weights/best.pt',
        #                             #'../kfolds/original_diopsis_data_set_train_val_synthetic_yolo/original_diopsis_data_set_train_val_synthetic_2_yolo/yolov10_run_original_diopsis_data_set_train_val_synthetic_2_yolo/yolov10n_b16_70e_original_diopsis_data_set_train_val_synthetic_2_yolo/weights/best.pt'
        #                            ],
        #     'average_weights':[0.2,0.2,0.2,0.2,0.2],
        #     'conf':0.25, 
        #     'image_size':[640,640],
        #     'remove_metrics_folder':True
        # },
      
        # {
        #     'yolo_model_name':'best.pt', 
        #     'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_synthetic_coco', 
        #     #'methods' :['sahi'], 
        #     #'methods' :['default','sahi'], 
        #     #'methods' :['default'], 
        #     'methods' :['ensemble'], 
        #     'conf':0.25, 
        #     'image_size':[640,640],
        #     'remove_metrics_folder':False
        # },

        {
            'yolo_model_name':'best.pt', 
            'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_yolo', 
            'methods' :['default','sahi'], 
            #'methods' :['default'], 
            'conf':0.25, 
            'image_size':[640,640],
            'remove_metrics_folder':True
        },
        {
            'yolo_model_name':'best.pt', 
            'yolo_root_path'  :'../kfolds/original_diopsis_data_set_train_val_synthetic_yolo', 
            'methods' :['default','sahi'], 
            #'methods' :['default'], 
            'conf':0.25, 
            'image_size':[640,640],
            'remove_metrics_folder':True
        },
    ]

# Get list of images
test_image_list_names=glob.glob('../original_diopsis_data_set_test/images/*.jpg')
#test_image_list_names=glob.glob('../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_0_yolo/images/val/*.jpg')

for yolo_dict in yolo_dicts:
    for method in yolo_dict['methods']:
        # Get subfolders of root folder
        sub_folders = [name for name in os.listdir(yolo_dict['yolo_root_path']) if os.path.isdir(os.path.join(yolo_dict['yolo_root_path'], name))]
        ################################## 
        for sub_folder in sub_folders:

            folder_metrics =[]
            if not os.path.exists(os.path.join(yolo_dict['yolo_root_path'],sub_folder,f"yolov10_run_{sub_folder}")):
                continue

            sub_sub_folders = [name for name in os.listdir(os.path.join(yolo_dict['yolo_root_path'],sub_folder,f"yolov10_run_{sub_folder}"))]
            if not method == 'average':
                # Load Yolo Model 
                yolo_model=os.path.join(yolo_dict['yolo_root_path'],sub_folder,f"yolov10_run_{sub_folder}",sub_sub_folders[0],'weights',yolo_dict['yolo_model_name'])
                if not os.path.exists(yolo_model):
                    continue
                print(yolo_model)
            
            try:
                del model
            except:
                pass
            torch.cuda.empty_cache()
            if method == 'ensemble':
                models_info=[
                            {'model_path': yolo_model,
                            'model':YOLOv10(yolo_model),
                            'image_size':[640,640],
                            'method':'default'},
                            {'model_path': yolo_model,
                            'model':YOLOv10(yolo_model),
                            'image_size':[800,800], 
                            'method':'sahi'},
                            ]
                ensemble = inf_types.YOLOv8Ensemble(models_info,device=device) 
            elif method == 'average':
                models=[]
                weights=[]
                
                for average_yolo_model in yolo_dict['average_yolo_models']:
                    models.append(YOLOv10(average_yolo_model).to(device))
                
                model = inf_types.average_yolov8_models(models,yolo_dict['average_weights'],device=device,)
            else:
                model = YOLOv10(yolo_model).to(device)

            metrics_folder= os.path.join(yolo_dict['yolo_root_path'],sub_folder,f"metrics_{yolo_dict['conf']}",method)
            metrics_file=os.path.join(metrics_folder,f"method_{method}_conf_{yolo_dict['conf']}_metrics.csv")
            print('Metrics file :',metrics_file)

            if (yolo_dict['remove_metrics_folder']==True) or (yolo_dict['remove_metrics_folder']==False and not os.path.exists(metrics_folder)):
                
                #Create metrics folder
                shutil.rmtree(metrics_folder,ignore_errors=True)
                tools.create_folders(metrics_folder)

                metrics_images_folder= os.path.join(metrics_folder,'images')
                tools.create_folders(metrics_images_folder)
            
                #create metrics file
                metrics=['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large', 'mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large', 'map_per_class', 'mar_100_per_class', 'classes']
                metrcs_labels_str=f"image,Pred_boxes,Target_boxes,{','.join(metrics)}"
                with open(metrics_file, 'w') as f:
                    f.write(metrcs_labels_str)
                    f.write('\n')
            
                # Load and predicti images    
                image_count=0
                for image_file_name in test_image_list_names:
                    pred_boxes=[]
                    pred_scores=[]
                    pred_labels=[]
                    target_boxes=[]
                    target_scores=[]
                    target_labels=[]
                    # Make prediction     
                    if method == 'ensemble':
                        results = ensemble.predict(image_file_name,conf=0.25, iou=0.7)#0.6
                    else:
                        results,original_image=inf_types.make_pedicts(model=model,
                                                        image_file_name=image_file_name,
                                                        device=device,
                                                        conf=yolo_dict['conf'],
                                                        image_size=yolo_dict['image_size'],
                                                        method=method
                                                        )
                
                    # Create Lists for metrics
                    yolo_tests_annotation_file=os.path.join('../original_diopsis_data_set_test/','labels',os.path.basename(image_file_name).replace('.jpg','.txt'))
                    #yolo_tests_annotation_file=os.path.join('../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo/original_diopsis_data_set_train_val_synthetic_cropted_0_yolo','labels','val',os.path.basename(image_file_name).replace('.jpg','.txt'))
                    list_boxes_annotation=list_boxes.get_list_box_yolo(yolo_tests_annotation_file,image_file_name)
                    for box in list_boxes_annotation['boxes']:
                            target_labels.append(int(box['object']))
                            target_scores.append(1)
                            target_boxes.append([ box['x1'],box['y1'],box['x2'],box['y2']]) 
                            #target_boxes.append([ box['cx'],box['cy'],box['w'],box['h']]) 
                    targets = [
                                    {
                                        'boxes': torch.tensor(target_boxes).to(torch.float).to(device),
                                        
                                        'labels': torch.tensor(target_labels).to(device)
                                    }
                                ]
                    if  method == 'default' or method == 'average':
                        #for boxes in results[0].boxes.xyxyn.data:    
                        #    scale_boxes.append([boxes[0]*original_image.shape[1],boxes[1]*original_image.shape[0],boxes[2]*original_image.shape[1],boxes[3]*original_image.shape[0]])
                        preds = [
                                    {
                                        'boxes':  torch.tensor(results[0].boxes.xyxy).to(device),
                                        #'boxes':  torch.tensor(results[0].boxes.cxcywh).to(device),
                                        'scores': torch.tensor(results[0].boxes.conf).to(device),
                                        'labels': torch.tensor(results[0].boxes.cls).to(torch.int).to(device)
                                    }
                                ]

                     
                    elif  method == 'sahi':
                        #tmp_results=results[results.confidence >= float(yolo_dict['conf'])]
                        results_confidence=[]
                        results_xyxy=[]
                        results_class_id=[]
                        for index in range(len(results.confidence)):
                            if results.confidence[index] >= float(yolo_dict['conf']):
                                results_confidence.append(results.confidence[index])
                                results_xyxy.append(results.xyxy[index])
                                results_class_id.append(results.class_id[index])
                                
                        preds = [
                                {
                                    'boxes':  torch.tensor(results_xyxy).to(device),
                                    'scores': torch.tensor(results_confidence).to(device),
                                    'labels': torch.tensor(results_class_id).to(torch.int).to(device)
                                }
                            ]

                        
                    elif  method == 'ensemble':
                        results_confidence=[]
                        results_xyxy=[]
                        results_class_id=[]
                        if results is not None:
                            for box in results:
                                x1, y1, x2, y2, score, cls = box
                                results_confidence.append(score)
                                results_xyxy.append((x1, y1, x2, y2))
                                results_class_id.append(cls)
                                    
                            preds = [
                                    {
                                        'boxes':  torch.tensor(results_xyxy).to(device),
                                        'scores': torch.tensor(results_confidence).to(device),
                                        'labels': torch.tensor(results_class_id).to(torch.int).to(device)
                                    }
                                ]

                    # Filtre Confidence
                    conf_threshold = float(yolo_dict['conf'])

                    filtered_preds = [
                                        {
                                            'boxes': p['boxes'][p['scores'] >= conf_threshold],
                                            'scores': p['scores'][p['scores'] >= conf_threshold],
                                            'labels': p['labels'][p['scores'] >= conf_threshold]
                                        }
                                        for p in preds
                                    ]            
                    preds = filtered_preds

                    # Save metrics
                    metrcs_str=get_metrics_str_for_csv(image_file_name,1,preds=preds,targets=targets,device=device)
                    
                    extra_str_tmp=metrcs_str.split(',')
                    extra_str=f"mAp:{extra_str_tmp[3]} mAp_50:{extra_str_tmp[4]} mAp_75:{extra_str_tmp[5]}"
                    with open(metrics_file, 'a') as f:
                        f.write(metrcs_str)
                        f.write('\n')

                    if image_count % 10 == 0:
                        tools.draw_bound_box_preds_targets_metrics(image_file_name,metrics_images_folder,preds,targets,extra_str)
                    image_count +=1    

            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                row_count = len(df)
                
                #condition = (df['Pred_boxes'] == 0) & (df['Target_boxes'] == 0)
                #df.loc[condition, df.columns.difference(['Pred_boxes', 'Target_boxes'])] = 1
                condition = (df['Pred_boxes'] == 0) & (df['Target_boxes'] == 0)
                mAP_columns = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large']
                df.loc[condition, mAP_columns] = 1  # Consider all mAP metrics as perfect for empty images

                
                condition2 = ((df['map']  == -1) & (df['map_50']  == -1) & (df['map_75']  == -1))
                #df.loc[condition2, df.columns.difference(['Pred_boxes', 'Target_boxes'])] = 0
                df = df[~condition2]
                
                #df.replace(-1, 0, inplace=True)
                df = df.replace(-1, np.nan)
                
                df = df.drop(['image','Pred_boxes','Target_boxes'], axis=1)
                df = df.mean()
                df = df.to_frame(name='Average').transpose()
                df.insert(0, 'folder_name', sub_folder)
                df.insert(1, 'Num_of_images', row_count)
                float_cols = df.select_dtypes(include=['float64','float32','float16','float']).columns
                df[float_cols] = df[float_cols].round(3)
                print(df)
                sum_metrics_file=os.path.join(yolo_dict['yolo_root_path'],f"method_{method}_conf_{yolo_dict['conf']}_metrics.csv")
                if os.path.exists(sum_metrics_file):
                    df_exist = pd.read_csv(sum_metrics_file)
                    df_exist = pd.concat([df_exist, df], ignore_index=True)
                    df_exist.to_csv(sum_metrics_file, index=False)
                else:
                    df_exist = df
                    df_exist.to_csv(sum_metrics_file, index=False)
                
                
            
        

