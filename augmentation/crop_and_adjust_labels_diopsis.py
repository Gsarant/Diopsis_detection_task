import cv2
import numpy as np
import os, sys
sys.path.append('./misc')
sys.path.append('./')
import tools
import glob 
import random
import json
import list_boxes

def crop_image_and_adjust_labels_diopsis(image_path, 
                                 label_path,
                                 new_diopsis_images_path,
                                 new_diopsis_annotations_path,
                                 x1, y1, x2, y2, 
                                 null_images=False):
    
    image = cv2.imread(str(image_path))
    boxes=list_boxes.get_list_box_diopsis(label_path, image_path)
    
    cropped_image = image[y1:y2, x1:x2]
    new_annotations=[]        
    for box in boxes['boxes']:
        
        class_id,probability, objectx1, objecty1, objectx2, objecty2 = box['object'],box['probability'],box['x1'],box['y1'],box['x2'],box['y2']
        
        # Υπολογισμός των νέων συντεταγμένων
        new_x1 = max(0, objectx1 - x1)
        new_y1 = max(0, objecty1 - y1)
        new_x2 = min(x2 - x1, objectx2 - x1)
        new_y2 = min(y2 - y1, objecty2 - y1)
        
        if new_x2 > new_x1 and new_y2 > new_y1:
            box1_area = (new_x2 - new_x1) * (new_y2 - new_y1)
            box2_area = (objectx2 -objectx1) * (objecty2 - objecty1)
            box_rate = min(box1_area,box2_area)/max(box1_area,box2_area)
            if box_rate > 0.6:
                new_annotations.append({
                    "labels": [
                        {
                            "probability": probability,
                            "name": class_id,
                            "color": f"#{random.randint(0, 0xFFFFFF):06x}ff"  # Τυχαίο χρώμα
                        }
                    ],
                    "shape": {
                        "x": int(new_x1),
                        "y": int(new_y1),
                        "width": int(new_x2 - new_x1),
                        "height": int(new_y2 - new_y1),
                        "type": "RECTANGLE"
                    }
                })
    
    # Δημιουργία του τελικού JSON
    json_data = {
        "annotations": new_annotations
    }
    
    if null_images==True or len(new_annotations) > 0:
        new_image_path = os.path.join(new_diopsis_images_path, f"cropped_{x2-x1}_{os.path.basename(image_path)}")
        new_label_path = os.path.join(new_diopsis_annotations_path, f"cropped_{x2-x1}_{os.path.basename(label_path)}")
    
        cv2.imwrite(str(new_image_path), cropped_image)
        with open(new_label_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        return new_image_path, new_label_path
    else:
        return None,None
    
    
def get_image_crop(image_path,crop_size=640):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return None
    max_x = w - crop_size
    max_y = h - crop_size
    x1 = random.randint(0, max_x)
    y1 = random.randint(0, max_y)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return x1, y1, x2, y2


def create_croped_images_diopsis(diopsis_images_path, 
                         diopsis_annotation_path,
                         new_diopsis_images_path,
                         new_diopsis_annotations_path,
                         test_images_path=None,
                         percentage_percent_number_of_images=0.1,
                         crop_size=640, 
                         null_images=False):
    
    tools.create_folders(new_diopsis_images_path)
    tools.create_folders(new_diopsis_annotations_path)
    
    diopsis_annotation_files_in_path=glob.glob(os.path.join(diopsis_annotation_path,'*.json'))
    random.shuffle(diopsis_annotation_files_in_path)
    number_of_images=int(percentage_percent_number_of_images*len(diopsis_annotation_files_in_path))
    index=0
    count=0
    while count<number_of_images :
        diopsis_annotation_file=diopsis_annotation_files_in_path[index]
        if os.stat(diopsis_annotation_file).st_size > 50 or null_images==True:
            diopsis_image_file=os.path.join(diopsis_images_path,os.path.basename(diopsis_annotation_file).replace('.json','.jpg'))
            x1, y1, x2, y2 = get_image_crop(diopsis_image_file,crop_size)
            if x1 is not None:
                new_image, new_labels= crop_image_and_adjust_labels_diopsis(diopsis_image_file,
                                                                     diopsis_annotation_file,
                                                                     new_diopsis_images_path,   
                                                                     new_diopsis_annotations_path,
                                                                     x1, y1, x2, y2)
                if new_image is not None:
                    if test_images_path is not None:
                        if count%50==0:
                            tools.create_folders(test_images_path)
                            tools.draw_bound_box_diopsis(new_image,  test_images_path)
                        print(count,index,new_image)
                    count +=1
        if index<len(diopsis_annotation_files_in_path)-1:
            index+=1
        else:
            index=0
            
        

if __name__ == '__main__':

    diopsis_images_path='../original_diopsis_data_set_train_val_synthetic/images'
    diopsis_annotation_path='../original_diopsis_data_set_train_val_synthetic/annotations'
    new_diopsis_images_path='../original_diopsis_data_set_train_val_synthetic_cropted/images'
    new_diopsis_annotations_path='../original_diopsis_data_set_train_val_synthetic_cropted/annotations'
   
    test_images_path='../original_diopsis_data_set_train_val_synthetic_cropted/test_images'
    crop_sizes=[480,640,720,1280]
    # for crop_size in crop_sizes:
    #     create_croped_images_diopsis(diopsis_images_path, 
    #                          diopsis_annotation_path,
    #                          new_diopsis_images_path,
    #                          new_diopsis_annotations_path,
    #                          test_images_path,
    #                          percentage_percent_number_of_images=0.2,
    #                          crop_size=crop_size, 
    #                          null_images=False)
    
    tools.copy_links(os.path.abspath(diopsis_images_path),os.path.abspath(new_diopsis_images_path))
    tools.copy_links(os.path.abspath(diopsis_annotation_path),os.path.abspath(new_diopsis_annotations_path))    