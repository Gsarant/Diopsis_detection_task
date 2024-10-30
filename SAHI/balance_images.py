import os, sys
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import glob
import shutil
import numpy as np
if __name__ == "__main__":
    dict_folders=[
     #   { 'coco_folder':'../kfolds/original_diopsis_data_set_train_val_coco'},
        {'coco_folder':'../kfolds/original_diopsis_data_set_train_val_synthetic_coco'},
    ]
    names=['Object']
    for dict_folder in dict_folders:
        sub_folders = [name for name in os.listdir(dict_folder['coco_folder']) if os.path.isdir(os.path.join(dict_folder['coco_folder'], name))]
        for sub_folder in sub_folders:
            coco_folder=os.path.join(dict_folder['coco_folder'],sub_folder)
            fullpath_output_coco_annotation_file_name=os.path.join(dict_folder['coco_folder'],sub_folder,'data.yml')
            coco_folder=[
                            {'type':'train','path':os.path.join(coco_folder,'train')},
                            {'type':'val','path':os.path.join(coco_folder,'val')}
                        ]
            for folder in coco_folder:
                null_images=[]
                not_null_images=[]
                for yolo_annotation_file in glob.glob(os.path.join(folder['path'],'*.txt')):
                    with open(yolo_annotation_file,'r') as f:
                        if  len(f.readlines())>0:
                            not_null_images.append(os.path.basename(yolo_annotation_file).split('.')[0])
                        else:
                            null_images.append(os.path.basename(yolo_annotation_file).split('.')[0])
                null_images=np.array(null_images)       
                np.random.shuffle(null_images)    
                if folder['type']=='train':
                    a= len(null_images)-int(60000-len(not_null_images))
                else:
                    a= len(null_images)-int(12000-len(not_null_images))
                print(os.path.join(folder['path']))        
                print('Null images',len(null_images))
                print('Not Null images',len(not_null_images))
                print('Sum',len(null_images)+len(not_null_images))
                print(a)
                print(len(not_null_images)+ (len(null_images)-a))        
                # null_folder=folder['path']+'NULL'
                # tools.create_folders(null_folder) 
                # for null_image in null_images[:a]:
                #     move_image_from=os.path.join(folder['path'],f"{null_image}.png")
                #     move_image_to=os.path.join(null_folder,f"{null_image}.png")
                #     move_annotation_from=os.path.join(folder['path'],f"{null_image}.txt")
                #     move_annotation_to=os.path.join(null_folder,f"{null_image}.txt")
                #     if os.path.exists(move_image_from) and os.path.exists(move_annotation_from):
                #         shutil.move(move_image_from,move_image_to)
                #         shutil.move(move_annotation_from,move_annotation_to)

