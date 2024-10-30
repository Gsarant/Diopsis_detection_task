from sahi.slicing import slice_coco
from sahi.utils.coco import Coco
from sahi.utils.file import save_json
import os, sys
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import glob
import shutil

def split_coco(coco_annotation_file_path, 
               image_dir, 
               output_coco_annotation_file_name,
               output_coco_path,
               slice_height=640, 
               slice_width=640, 
               overlap_height_ratio=0.2, 
               overlap_width_ratio=0.2
               ):
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=image_dir,
        output_coco_annotation_file_name=output_coco_annotation_file_name,
        output_dir=output_coco_path,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        
    )
    print('Create annotation file ',output_coco_annotation_file_name)

    
   
    
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
                   
            if os.path.exists(fullpath_output_coco_annotation_file_name):
                continue
            coco_folder=[
                            {'type':'train','path':os.path.join(coco_folder,'train')},
                            {'type':'val','path':os.path.join(coco_folder,'val')}
                        ]
            for folder in coco_folder:
                count =0 
                for coco_annotation_file in glob.glob(os.path.join(folder['path'],'*.json')):
                    output_coco_annotation_file_name=os.path.join("../",f"{folder['type']}_split_{os.path.basename(coco_annotation_file).split('.')[0]}")
                    output_coco_path = os.path.join(folder['path'],'split') 
                    shutil.rmtree(output_coco_path, ignore_errors=True)
                    print('Delete Folder',output_coco_path)
                    tools.create_folders(output_coco_path)    
                    print('Create Folder',output_coco_path)
                    #if os.path.exists(output_coco_annotation_file_name):
                    #   os.remove(output_coco_annotation_file_name)
                    #   print('Remove File',output_coco_annotation_file_name)
                   
                    print('output_coco_annotation_file_name ',output_coco_annotation_file_name)    
                    split_coco(
                            coco_annotation_file_path=coco_annotation_file,
                            image_dir='',
                            output_coco_path = output_coco_path,
                            output_coco_annotation_file_name=output_coco_annotation_file_name
                            )
            # yolo_folder=os.path.join(os.path.abspath(dict_folder['coco_folder']),sub_folder)
            # print('yolo_folder',yolo_folder)
            # yolo_yalm_file=os.path.join(yolo_folder,'data.yml')   
            # print(yolo_yalm_file)
            # print('***',glob.glob(os.path.join(yolo_folder,'train','*split_annotations*.json'))[0])
            # with open(yolo_yalm_file,'x') as f:
            #     f.write(f"names: {names}\n")
            #     f.write("nc: 1\n")
            #     f.write(f"train: {os.path.join(os.path.abspath(yolo_folder),'train','split')}\n")
            #     f.write(f"val: {os.path.join(os.path.abspath(yolo_folder),'val','split')}\n")
            #     f.write(f"train_ann: {glob.glob(os.path.join(yolo_folder,'train','*split_annotations*.json'))[0]}\n")
            #     f.write(f"val_ann: {glob.glob(os.path.join(yolo_folder,'val','*split_annotations*.json'))[0]}\n")
            #     #f.write("augmentations:\n")
            #     #f.write(" fliplr: 0.5\n")  # 50% chance to flip images left-right
            #     #f.write(" flipud: 0.2\n")  # 20% chance to flip images up-down
            #     #f.write(" scale: 0.1\n")   # Scale images by +/- 10%
            #     #f.write(" translate: 0.1\n")  # Translate images by +/- 10% of image dimensions
            #     #f.write(" rotate: 5\n")  # Rotate images by +/- 5 degrees
            #     #f.write(" shear: 5\n")  # Shear images by +/- 5 degrees
            #     #f.write(" perspective: 0.05\n")  # Apply perspective transformation with a probability
            #     #f.write(" mosaic: 0.5\n")  # 75% chance to apply mosaic augmentation
            #     #f.write(" mixup: 0.4\n")  # 40% chance to apply mixup augmentation
         
