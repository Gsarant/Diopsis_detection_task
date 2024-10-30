import glob
import shutil
import os, sys
import cv2
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import list_boxes



def convert_diopsis_to_yolo(diopsis_folder,yolo_folder,list_of_names):
    diopsis_folder=[
        {'type':'train','path':os.path.join(diopsis_folder,'train')},
        {'type':'val','path':os.path.join(diopsis_folder,'val')}
    ]

    for folder in diopsis_folder:
        yolo_images_path=os.path.join(yolo_folder,'images')
        tools.create_folders(yolo_images_path)
        yolo_labels_path=os.path.join(yolo_folder,'labels')
        tools.create_folders(yolo_labels_path)

        yolo_images_train_path=os.path.join(yolo_images_path,'train')
        tools.create_folders(yolo_images_train_path)
        yolo_images_val_path=os.path.join(yolo_images_path,'val')
        tools.create_folders(yolo_images_val_path)
        
        yolo_labels_train_path=os.path.join(yolo_labels_path,'train')
        tools.create_folders(yolo_labels_train_path)
        yolo_labels_val_path=os.path.join(yolo_labels_path,'val')
        tools.create_folders(yolo_labels_val_path)

        count =0 
        for diopsis_annotation_file in glob.glob(os.path.join(folder['path'],'annotations','*.json')):
            
            diopsis_image_file=tools.convert_name_annotation_file_to_image_file(diopsis_annotation_file)
            yolo_image_file=os.path.join(yolo_images_path,folder['type'],os.path.basename(diopsis_image_file))
            
            image=cv2.imread(os.path.abspath(diopsis_image_file))
            img_w, img_h = image.shape[1],image.shape[0]

            tools.copy_symlink(os.path.abspath(diopsis_image_file),os.path.abspath(yolo_image_file))

            list_boxes_annotation=list_boxes.get_list_box_diopsis(diopsis_annotation_file,diopsis_image_file)
            yolo_annotation_file=os.path.join(yolo_labels_path,folder['type'],os.path.basename(diopsis_annotation_file).replace('.json','.txt'))
            with open(os.path.abspath(yolo_annotation_file),'a') as f:
                for box in list_boxes_annotation['boxes']:
                    cx = (box['x1']+box['x2'])/2
                    cy = (box['y1']+box['y2'])/2
                    w = (box['x2']-box['x1'])
                    h = (box['y2']-box['y1'])
                    f.write(f"{list_of_names.index(box['object'])} {(cx/img_w):.6f} {(cy/img_h):.6f} {(w/img_w):.6f} {(h/img_h):.6f} \n")
            count +=1
            if count%200==0:
                test_yolo_format(yolo_image_file,yolo_annotation_file,yolo_folder)

def test_yolo_format(yolo_image,yolo_annotation,yolo_folder):
    image=cv2.imread(yolo_image)
    image= cv2.resize(image, (640, 640))
    img_w, img_h = image.shape[1],image.shape[0]
    with open(yolo_annotation) as f:
        for line in f:
            box = line.strip().split(' ')
            #print(box)
            h = float(box[4])
            w = float(box[3])
            cx = float(box[1])
            cy = float(box[2])            
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            
            cv2.rectangle(image, (x1,y1), 
                                  (x2,y2), 
                                   (0, 255, 0), 2)
    yolo_annotated_image_folder=os.path.join(yolo_folder,'yolo_annotated_images_train')
    tools.create_folders(yolo_annotated_image_folder)
    yolo_annotated_image=os.path.join(yolo_annotated_image_folder, os.path.basename(yolo_image))
    
    cv2.imwrite(yolo_annotated_image,image)   

if __name__ == "__main__":
    dict_folders=[
        #{'diopsis_folder':'../kfolds/original_diopsis_data_set_train_val', 'yolo_folder': '../kfolds/original_diopsis_data_set_train_val_yolo'},
        #{'diopsis_folder':'../kfolds/original_diopsis_data_set_train_val_synthetic', 'yolo_folder': '../kfolds/original_diopsis_data_set_train_val_synthetic_yolo'}
        {'diopsis_folder':'../kfolds/original_diopsis_data_set_train_val_synthetic_cropted', 'yolo_folder': '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo'}
    ]
    names=['Object']
    for dict_folder in dict_folders:
        sub_folders = [name for name in os.listdir(dict_folder['diopsis_folder']) if os.path.isdir(os.path.join(dict_folder['diopsis_folder'], name))]
        for sub_folder in sub_folders:
            diopsis_folder=os.path.join(dict_folder['diopsis_folder'],sub_folder)
            yolo_folder=os.path.join(dict_folder['yolo_folder'],f"{sub_folder}_yolo")
            print (diopsis_folder, os.path.isdir(diopsis_folder))
            print (yolo_folder)
            convert_diopsis_to_yolo(diopsis_folder,yolo_folder,names)         
            with open(f"{yolo_folder}/data.yml",'x') as f:
                f.write(f"names: {names}\n")
                f.write("nc: 1\n")
                f.write(f"train: {os.path.join(os.path.abspath(yolo_folder),'images','train')}\n")
                f.write(f"val: {os.path.join(os.path.abspath(yolo_folder),'images','val')}\n")
                #f.write("augmentations:\n")
                #f.write(" fliplr: 0.5\n")  # 50% chance to flip images left-right
                #f.write(" flipud: 0.2\n")  # 20% chance to flip images up-down
                #f.write(" scale: 0.1\n")   # Scale images by +/- 10%
                #f.write(" translate: 0.1\n")  # Translate images by +/- 10% of image dimensions
                #f.write(" rotate: 5\n")  # Rotate images by +/- 5 degrees
                #f.write(" shear: 5\n")  # Shear images by +/- 5 degrees
                #f.write(" perspective: 0.05\n")  # Apply perspective transformation with a probability
                #f.write(" mosaic: 0.5\n")  # 75% chance to apply mosaic augmentation
                #f.write(" mixup: 0.4\n")  # 40% chance to apply mixup augmentation
          


    #for yolo_image in glob.glob(os.path.join(yolo_folder,'train','*.jpg')):
    #    yolo_annotation=yolo_image.replace('.jpg','.txt')
    #    test_yolo_format(yolo_image,yolo_annotation,'./')
    