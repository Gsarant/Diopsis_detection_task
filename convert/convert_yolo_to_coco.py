import os,sys
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import json
from PIL import Image

def convert_yolo_to_coco(yolo_dir, output_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Assuming classes.txt exists in the yolo_dir
    with open(os.path.join(yolo_dir, "classes.txt"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    for i, cls in enumerate(classes):
        coco_format["categories"].append({
            "id": i + 1,
            "name": cls,
            "supercategory": "none"
        })
    
    annotation_id = 1
    for img_file in os.listdir(yolo_dir):
        if img_file.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(yolo_dir, img_file)
            img = Image.open(img_path)
            width, height = img.size
            
            image_id = len(coco_format["images"]) + 1
            coco_format["images"].append({
                "id": image_id,
                "file_name": img_file,
                "width": width,
                "height": height
            })
            
            # Read corresponding YOLO annotation file
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(yolo_dir, txt_file)
            
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, box_width, box_height = map(float, line.split())
                        
                        # Convert YOLO format to COCO format
                        x = (x_center - box_width / 2) * width
                        y = (y_center - box_height / 2) * height
                        w = box_width * width
                        h = box_height * height
                        
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id) + 1,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1
    
    # Save COCO format JSON
    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=2)

if __name__ == "__main__":
    folders_dicts=[
        {'root_folder_yolo':"../kfolds/original_diopsis_data_set_train_val_synthetic_yolo", 'root_folder_coco':"../kfolds/original_diopsis_data_set_train_val_synthetic_coco"},
    ]
    for folder_dict in folders_dicts:
        sub_folders = [name for name in os.listdir(folder_dict['root_folder_yolo']) if os.path.isdir(os.path.join(folder_dict['root_folder_yolo'], name))]
        for sub_folder in sub_folders:
            yolo_folder = os.path.join(folder_dict['root_folder_yolo'],sub_folder)
            coco_folder = os.path.join(folder_dict['root_folder_coco'],sub_folder.replace('yolo','coco'))
            #tools.create_folders(coco_folder)
            print('yolo_folder',yolo_folder,os.path.exists(yolo_folder))
            print('coco_folder',coco_folder,'\n')
            convert_yolo_to_coco(yolo_folder,coco_folder)

