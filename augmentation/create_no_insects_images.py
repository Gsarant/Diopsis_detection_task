
import cv2 as cv
import glob 
import numpy as np
from create_background_images import create_no_insect_images
import list_boxes
import shutil
import os, sys
sys.path.append('./misc')
sys.path.append('./')
import tools

# def get_image_and_boxes_per_image_from_list_box(image,annotation_boxes_coords):
#     list_of_insects_images_per_image=[]
#     for annotation_box_coords in annotation_boxes_coords:
#         x1 = int(annotation_box_coords['x1'])
#         y1 = int(annotation_box_coords['y1'])
#         x2 = int(annotation_box_coords['x2'])
#         y2 = int(annotation_box_coords['y2'])
#         tmp_image=image[y1:y2, x1:x2]
#         list_of_insects_images_per_image.append({'image':tmp_image,'box':annotation_box_coords})
#     return list_of_insects_images_per_image

def remove_background(image):
    img=image
   # Create a mask initialized with obvious background and foreground
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Create rectangular region for the foreground
    rect = (10, 10, img.shape[1]-20, img.shape[0]-20)
    
    # Create temporary arrays for GrabCut
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    # Run GrabCut
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)
    
    # Modify the mask
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Apply the mask to the image
    result = img * mask2[:,:,np.newaxis]
    
    # Create a white background
    white_background = np.ones_like(img, np.uint8) * 255
    white_background = white_background * (1 - mask2[:,:,np.newaxis])
    
    # Combine the result with the white background
    result = result + white_background
    return result
def paste( background_part, pasted_image):
        pasted_image_bg = cv.cvtColor(pasted_image, cv.COLOR_RGB2GRAY)
        background_part[:,:,0] = np.where( pasted_image_bg < 240, pasted_image[:,:,0], background_part[:,:,0])
        background_part[:,:,1] = np.where( pasted_image_bg < 240, pasted_image[:,:,1], background_part[:,:,1])
        background_part[:,:,2] = np.where( pasted_image_bg < 240, pasted_image[:,:,2], background_part[:,:,2])
            #150
        return background_part
def overlay_image(background_image, overlay_image, annotation_box_coords):
    x1_background = int(annotation_box_coords['x1'])
    x2_background = int(annotation_box_coords['x2'])
    y1_background = int(annotation_box_coords['y1'])
    y2_background = int(annotation_box_coords['y2'])
    #        background_part[:,:,0] = np.where( pasted_image_bg < 240, pasted_image[:,:,0], background_part[:,:,0])
    background_image[y1_background:y2_background, x1_background:x2_background, :] = paste(
            background_image[y1_background:y2_background, x1_background:x2_background, :], overlay_image)
   
    return background_image


def create_synthetic_image(background_image,original_image_path,list_box_per_image):
    insects_image = cv.imread(os.path.join(original_image_path,list_box_per_image['image']))
    for background_index,item_list_box_per_image in enumerate(list_box_per_image['boxes']):
        x1 = int(item_list_box_per_image['x1'])
        y1 = int(item_list_box_per_image['y1'])
        x2 = int(item_list_box_per_image['x2'])
        y2 = int(item_list_box_per_image['y2'])
        one_insect_image=insects_image[y1:y2, x1:x2]
        insect_image = remove_background(one_insect_image)
        background_image=overlay_image(background_image,insect_image,item_list_box_per_image)
    return background_image


    

        
def create_backgound_images_and_synthetic(original_path,synthetic_path,number_of_images_background,number_images_with_insects):
    
    #list of diopsis annotations
    list_of_diopsis_anotation_files=glob.glob(os.path.join(original_path,'annotations','*.json'))
    #create list_box type from list of diopsis annotations
    list_box_diopsis_annotations=[]
    for diopsis_annotation_file in list_of_diopsis_anotation_files:
        tmp_list_box=list_boxes.get_list_box_diopsis(annotation_file=diopsis_annotation_file,
                                    image_file=tools.convert_name_annotation_file_to_image_file(diopsis_annotation_file))
        if len(tmp_list_box['boxes'])>0:
            list_box_diopsis_annotations.append(tmp_list_box)
    #create list_box type with background files
    list_box_diopsis_annotations=np.array(list_box_diopsis_annotations)
    np.random.shuffle(list_box_diopsis_annotations)
    list_box_diopsis_annotations_background=list_box_diopsis_annotations[:number_of_images_background]
    #create list_box type with insect files
    list_box_diopsis_annotations_insects= [i for i in list_box_diopsis_annotations if i not in list_box_diopsis_annotations_background]
    np.random.shuffle(list_box_diopsis_annotations_insects)
    #create folder for background images in synthetic path
    background_images_path=os.path.join(synthetic_path,'background_images')
    tools.create_folders(background_images_path)
    #create background images to synthetic path background_images folder
    create_no_insect_images(list_box_annotations=list_box_diopsis_annotations_background,
                            original_images_path=os.path.join(original_path,'images'),
                            save_no_insects_images_path=background_images_path
                            )
    
    # create folder for synthetic images in synthetic path
    synthetic_images_path=os.path.join(synthetic_path,'images')
    tools.create_folders(synthetic_images_path)
    synthetic_annotation_path=os.path.join(synthetic_path,'annotations')
    tools.create_folders(synthetic_annotation_path)
    #create synthetic images from background images
    
    for background_image_path in glob.glob(os.path.join(background_images_path,'*.jpg')):
        tmp_background_image=cv.imread(background_image_path)
        for index,item_list_box_diopsis_annotations_insects in enumerate(list_box_diopsis_annotations_insects[:number_images_with_insects]): 
            background_image=tmp_background_image.copy()
            synthetic_image=create_synthetic_image(background_image=background_image,
                                   original_image_path=os.path.join(original_path,'images'),
                                   list_box_per_image=item_list_box_diopsis_annotations_insects
                                    )
            synthetic_image_path=os.path.join(synthetic_images_path,
                                              f"{os.path.basename(background_image_path).split('.')[0]}_{os.path.basename(item_list_box_diopsis_annotations_insects['image'].split('.')[0])}.jpg")
            cv.imwrite(synthetic_image_path,synthetic_image)
            original_annotation_path=os.path.join(original_path,'annotations',item_list_box_diopsis_annotations_insects['image'].replace('.jpg','.json'))
            synthetic_annotation_path=tools.convert_name_image_file_to_annotation_file(synthetic_image_path)
            shutil.copy(original_annotation_path,synthetic_annotation_path)
            # synthetic_test_path=os.path.join(synthetic_path,'test')
            # tools.create_folders(synthetic_test_path)
            # tools.draw_bound_box_diopsis(synthetic_image_path,synthetic_test_path)
 
if __name__ == '__main__':

    original_path='../original_diopsis_data_set_train_val'
    synthetic_path='../synthetic_diopsis_data_set_train_val'
    create_backgound_images_and_synthetic(original_path=original_path,
                                          synthetic_path=synthetic_path,
                                          number_of_images_background=150,
                                          number_images_with_insects=15
                                          )


  


