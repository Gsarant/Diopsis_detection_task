import glob
import shutil
import os, sys
import cv2
sys.path.append('./misc')
sys.path.append('./augmentation')
sys.path.append('./')
import tools
import list_boxes



def convert_diopsis_to_yolo(diopsis_folder,yolo_folder,images_folder,list_of_names):
    
    count =0 
    tools.create_folders(yolo_folder)

    for diopsis_annotation_file in glob.glob(os.path.join(diopsis_folder,'*.json')):
        diopsis_image_file=tools.convert_name_annotation_file_to_image_file(diopsis_annotation_file)
        yolo_image_file=os.path.join(images_folder,os.path.basename(diopsis_image_file))
        
        image=cv2.imread(diopsis_image_file)
        img_w, img_h = image.shape[1],image.shape[0]
        list_boxes_annotation=list_boxes.get_list_box_diopsis(diopsis_annotation_file,diopsis_image_file)
        yolo_annotation_file=os.path.join(yolo_folder,os.path.basename(diopsis_annotation_file).replace('.json','.txt'))
        with open(yolo_annotation_file,'a') as f:
            for box in list_boxes_annotation['boxes']:
                cx = (box['x1']+box['x2'])/2
                cy = (box['y1']+box['y2'])/2
                w = (box['x2']-box['x1'])
                h = (box['y2']-box['y1'])
                f.write(f"{list_of_names.index(box['object'])} {(cx/img_w):.6f} {(cy/img_h):.6f} {(w/img_w):.6f} {(h/img_h):.6f} \n")
        count +=1
        tools.draw_bound_box_yolo_format(yolo_image_file,yolo_annotation_file,yolo_folder,(640,640))



if __name__ == "__main__":
    names=['Object']
    diopsis_folder='../original_diopsis_data_set_test/annotations'
    yolo_folder='../original_diopsis_data_set_test/labels'
    images_folder='../original_diopsis_data_set_test/images'
    convert_diopsis_to_yolo(diopsis_folder,yolo_folder,images_folder,names)         
    with open(f"../original_diopsis_data_set_test/data.yml",'x') as f:
        f.write(f"names: {names}\n")
        f.write("nc: 1\n")
        #f.write(f"train: {os.path.join(os.path.abspath(yolo_folder),'images','train')}\n")
        #f.write(f"val: {os.path.join(os.path.abspath(yolo_folder),'images','val')}\n")
        f.write(f"test: {os.path.join(os.path.abspath(images_folder))}\n")
      