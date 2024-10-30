import glob
import shutil
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation,CocoPrediction
from sahi.utils.file import save_json
from sahi.utils.file import load_json
import os, sys
import cv2
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import list_boxes


def convert_diopsis_to_coco(diopsis_folder,coco_folder,list_of_names):
    diopsis_folder=[
            {'type':'train','path':os.path.join(diopsis_folder,'train')},
            {'type':'val','path':os.path.join(diopsis_folder,'val')}
        ]
    for folder in diopsis_folder:
        coco = Coco()
        for index,name in enumerate(list_of_names):
            coco.add_category(CocoCategory(id=index, name=name))
        
        count =0 
        for diopsis_annotation_file in glob.glob(os.path.join(folder['path'],'annotations','*.json')):
            #coco_images_path=os.path.join(coco_folder,folder['type'],'images')
            #tools.create_folders(coco_images_path)
            diopsis_image_file=tools.convert_name_annotation_file_to_image_file(diopsis_annotation_file)
            #coco_image_file = os.path.join(coco_images_path,os.path.basename(diopsis_image_file))
            # shutil.copy(diopsis_image_file,coco_image_file)
            image=cv2.imread(diopsis_image_file)
            img_w, img_h = image.shape[1],image.shape[0]
            list_boxes_annotation=list_boxes.get_list_box_diopsis(diopsis_annotation_file,diopsis_image_file)  
            coco_image = CocoImage(file_name=diopsis_image_file, height=img_h, width=img_w)
            for annotation in list_boxes_annotation['boxes']:
                coco_image.add_annotation(
                        CocoAnnotation(
                        bbox=[annotation['x1'], annotation['y1'], annotation['x2']-annotation['x1'], annotation['y2']-annotation['y1']],
                        category_id=list_of_names.index(annotation['object']),
                        category_name=annotation['object'],
                    )
                    )
            coco.add_image(coco_image)
            count += 1
        coco_file=os.path.join(coco_folder,folder['type'],f'annotations_{folder["type"]}_{len(coco.images)}.json')
        save_json(data=coco.json, save_path=coco_file)
        print('Create :' + coco_file)
        annotation_coco_images_path=os.path.join(coco_folder,folder['type'],'annotation_coco_images')
        tools.create_folders(annotation_coco_images_path)
        test(coco_file,annotation_coco_images_path)  

def test(coco_annotation_file,coco_images_path):
    coco = Coco()
    cocos=coco.from_coco_dict_or_path(coco_annotation_file)
    
    
    count =0 
    for coco_image in cocos.images:
        print(coco_image.file_name)
        img = cv2.imread(coco_image.file_name)
        
        for annotations in coco_image.annotations:
            
            xywh = annotations.bbox
            xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
            
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        count += 1
       
        annotation_coco_image_file=os.path.join(coco_images_path,os.path.basename(coco_image.file_name))
        cv2.imwrite(annotation_coco_image_file, img)
        print('Create annotation image ',annotation_coco_image_file)
        if count > 10:
            break

if __name__ == "__main__":
    dict_folders=[
     #   {'diopsis_folder':'../kfolds/original_diopsis_data_set_train_val', 'coco_folder':'../kfolds/original_diopsis_data_set_train_val_coco'},
        {'diopsis_folder':'../kfolds/original_diopsis_data_set_train_val_synthetic', 'coco_folder':'../kfolds/original_diopsis_data_set_train_val_synthetic_coco'},
    ]
    names=['Object']
    for dict_folder in dict_folders:
        sub_folders = [name for name in os.listdir(dict_folder['diopsis_folder']) if os.path.isdir(os.path.join(dict_folder['diopsis_folder'], name))]
        for sub_folder in sub_folders:
            diopsis_folder=os.path.join(dict_folder['diopsis_folder'],sub_folder)
            coco_folder=os.path.join(dict_folder['coco_folder'],sub_folder)
            # shutil.rmtree(coco_folder, ignore_errors=True)
            # print('Delete ',coco_folder)
            if not os.path.exists(coco_folder):
                tools.create_folders(coco_folder)
                print('Create Folder',coco_folder)
                convert_diopsis_to_coco(diopsis_folder,coco_folder,names)
            else:
                continue