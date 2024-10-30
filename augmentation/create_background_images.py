import numpy as np
import cv2 as cv
import os

def remove_box_and_inpaint(image, annotation_box_coords):
    x1 = int(annotation_box_coords['x1'])
    y1 = int(annotation_box_coords['y1'])
    x2 = int(annotation_box_coords['x2'])
    y2 = int(annotation_box_coords['y2'])
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    result = cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
    return result

def create_no_insect_images(list_box_annotations,original_images_path='',save_no_insects_images_path='',save_original_images_path=None):
    for index,item_of_list_image in enumerate(list_box_annotations):
        image=cv.imread(os.path.join(original_images_path,item_of_list_image['image']))
        #if save_original_images_path is not None:
        #    cv.imwrite(os.path.join(save_original_images_path,os.path.basename(item_of_list_image['image'])),image)
        for annotation_box in item_of_list_image['boxes']:
            image=remove_box_and_inpaint(image=image,annotation_box_coords=annotation_box)
        cv.imwrite(os.path.join(save_no_insects_images_path,os.path.basename(item_of_list_image['image'])),image)
       