import os,sys
import cv2
import ultralytics
from ultralytics import YOLOv10
import torch
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools

urls = [
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt"
]

class Yolov10train:
    def __init__(self,yolo_model_name,yaml_path,output_path,name,image_size=640,batch=16,epochs=70):
        self.yolo_model_name=yolo_model_name
        self.yaml_path=yaml_path
        self.output_path=output_path
        self.name=name
        self.epochs=epochs
        self.batch=batch
        self.image_size=image_size
    def train(self):        
        model = YOLOv10(self.yolo_model_name)
        
        results = model.train(
            data= self.yaml_path,         # Path to your dataset config file
            batch = self.batch,               # Training batch size
            imgsz= self.image_size,                   # Input image size
            epochs= self.epochs,                  # Number of training epochs
            #optimizer= 'SGD',             # Optimizer, can be 'Adam', 'SGD', etc.
            # optimizer= 'Adam',             # Optimizer, can be 'Adam', 'SGD', etc.
            # lr0= 0.01,                    # Initial learning rate SGD
            # lr0= 0.001,                    # Initial learning rate Adam
            # lrf= 0.1,                     # Final learning rate factor
            # weight_decay= 0.0005,         # Weight decay for regularization SGD
            # momentum= 0.937,              # Momentum (SGD-specific) SGD
            verbose= True,                # Verbose output
            device= '0',                  # GPU device index or 'cpu'
            workers= 8,                   # Number of workers for data loading
            project= self.output_path,        # Output directory for results
            name= self.name,                  # Experiment name
            exist_ok= False,              # Overwrite existing project/name directory
            #rect= True,                  # Use rectangular training (speed optimization)
            #resume= False,                # Resume training from the last checkpoint
            #multi_scale= False,           # Use multi-scale training
            single_cls= False,             # Treat data as single-class
            #amp=True,
            #cache=True,
            #patience=5,
        )

def train():
    
    yolo_dicts=[
        #{'yolo_model_name':'weights/yolov10n.pt', 'yolo_root_path' :'../kfolds/original_diopsis_data_set_train_val_yolo', 'epochs':70 , 'batch':16 , 'image_size':640},
        #{'yolo_model_name':'weights/yolov10n.pt', 'yolo_root_path' :'../kfolds/original_diopsis_data_set_train_val_synthetic_yolo' , 'epochs':70 , 'batch':16 , 'image_size':640},
        #{'yolo_model_name':'weights/yolov10n.pt', 'yolo_root_path' :'../kfolds/original_diopsis_data_set_train_val_synthetic_coco' , 'epochs':50 , 'batch':-1 , 'image_size':320},
        {'yolo_model_name':'weights/yolov10n.pt', 'yolo_root_path' :'../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo' , 'epochs':70 , 'batch':-1 , 'image_size':640},
    ]
    for yolo_dict in yolo_dicts:
        sub_folders = [name for name in os.listdir(yolo_dict['yolo_root_path']) if os.path.isdir(os.path.join(yolo_dict['yolo_root_path'], name))]
        for sub_folder in sub_folders:
            torch.cuda.empty_cache()
            yaml_path=os.path.join(os.path.abspath(yolo_dict['yolo_root_path']),sub_folder,'data.yml')
            output_path=os.path.join(os.path.abspath(yolo_dict['yolo_root_path']),sub_folder,f"yolov10_run_{sub_folder}")
            print('yaml_path',yaml_path)
            print('output_path',output_path)
            if os.path.exists(yaml_path):
                if not os.path.exists(output_path):
                    tools.create_folders(output_path)
                    name=f"yolov10n_b{yolo_dict['batch']}_{yolo_dict['epochs']}e_{sub_folder}"
                    print('yaml_path',yaml_path)
                    print('output_path',output_path)
                    print('name',name)
                    yolotrain=Yolov10train(yolo_model_name=yolo_dict['yolo_model_name'],
                                        yaml_path=yaml_path,
                                        output_path=output_path,
                                        name=name,
                                        image_size=yolo_dict['image_size'],
                                        epochs=yolo_dict['epochs'],
                                        batch=yolo_dict['batch']
                                        ).train()
                    print('Finished Training',yaml_path)

            

if __name__ == "__main__":
    train()