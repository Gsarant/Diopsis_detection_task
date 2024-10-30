import cv2
import numpy as np
import os, sys
sys.path.append('./misc')
sys.path.append('./')
import tools
import glob 
import random

def crop_image_and_adjust_labels(image_path, 
                                 label_path,
                                 new_yolo_images_path, 
                                 new_yolo_labels_path, 
                                 x1, y1, x2, y2, 
                                 null_images=False):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    
    cropped_image = image[y1:y2, x1:x2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x_center, y_center = x_center * w, y_center * h
        width, height = width * w, height * h
        new_x_center = x_center - x1
        new_y_center = y_center - y1
        if (0 < new_x_center < (x2-x1) and 0 < new_y_center < (y2-y1) and
            new_x_center - width/2 < (x2-x1) and new_x_center + width/2 > 0 and
            new_y_center - height/2 < (y2-y1) and new_y_center + height/2 > 0):
    
            new_width = min(width, 2 * min(new_x_center, (x2-x1) - new_x_center))
            new_height = min(height, 2 * min(new_y_center, (y2-y1) - new_y_center))
    
            new_x_center /= (x2 - x1)
            new_y_center /= (y2 - y1)
            new_width /= (x2 - x1)
            new_height /= (y2 - y1)
            
            new_lines.append(f"{int(class_id)} {new_x_center} {new_y_center} {new_width} {new_height}\n")
    
    if null_images==True or len(new_lines) > 0:
        new_image_path = os.path.join(new_yolo_images_path, f"cropped_{x2-x1}_{os.path.basename(image_path)}")
        new_label_path = os.path.join(new_yolo_labels_path, f"cropped_{x2-x1}_{os.path.basename(label_path)}")
    
        cv2.imwrite(str(new_image_path), cropped_image)
        with open(new_label_path, 'w') as f:
            f.writelines(new_lines)
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


def create_croped_images(yolo_images_path, 
                         yolo_labels_path,new_yolo_images_path, 
                         new_yolo_labels_path,test_images_path=None,
                         percentage_percent_number_of_images=0.1,
                         crop_size=640, 
                         null_images=False):
    
    tools.create_folders(new_yolo_images_path)
    tools.create_folders(new_yolo_labels_path)
    yolo_annotation_files_in_path=glob.glob(os.path.join(yolo_labels_path,'*.txt'))
    random.shuffle(yolo_annotation_files_in_path)
    number_of_images=int(percentage_percent_number_of_images*len(yolo_annotation_files_in_path))
    index=0
    count=0
    while count<number_of_images and index<len(yolo_annotation_files_in_path):
        yolo_annotation_file=yolo_annotation_files_in_path[index]
        if os.stat(yolo_annotation_file).st_size > 0 or null_images==True:
            yolo_image_file=os.path.join(yolo_images_path,os.path.basename(yolo_annotation_file).replace('.txt','.jpg'))
            x1, y1, x2, y2 = get_image_crop(yolo_image_file,crop_size)
            if x1 is not None:
                new_image, new_labels= crop_image_and_adjust_labels(yolo_image_file, 
                                                                     yolo_annotation_file,
                                                                     new_yolo_images_path, 
                                                                     new_yolo_labels_path, 
                                                                     x1, y1, x2, y2)
                if new_image is not None:
                    if test_images_path is not None:
                        tools.draw_bound_box_yolo_format(new_image, new_labels, test_images_path)
                        print(index,new_image)
                    count +=1
        index+=1
            
        

if __name__ == '__main__':

    yolo_images_path='../original_diopsis_data_set_test/images'
    yolo_labels_path='../original_diopsis_data_set_test/labels'
    new_yolo_images_path='../original_diopsis_data_set_test/images_cropped'
    new_yolo_labels_path='../original_diopsis_data_set_test/labels_cropped'
    test_images_path='../original_diopsis_data_set_test/test_images'
    crop_sizes=[480,640,1280]
    for crop_size in crop_sizes:
        create_croped_images(yolo_images_path, 
                             yolo_labels_path,
                             new_yolo_images_path, 
                             new_yolo_labels_path,
                             test_images_path,
                             percentage_percent_number_of_images=0.1,
                             crop_size=crop_size, 
                             null_images=False)