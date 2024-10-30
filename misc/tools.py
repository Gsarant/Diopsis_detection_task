import os
import cv2
import json

def create_folders(folders):
        if not os.path.isdir( folders):
            folders_array = folders.split(os.sep)
            for f in range(len(folders_array)):
                cur_folder = os.path.sep.join(folders_array[:(f+1)])
                try:
                    os.makedirs(cur_folder)
                except:
                    pass

def convert_name_annotation_file_to_image_file(annotation_file_path):
    path_with_out_last_folder=annotation_file_path.split(os.path.sep)[:-2]
    return os.path.join(*path_with_out_last_folder,'images',os.path.basename(annotation_file_path).replace('.json','.jpg'))

def convert_name_image_file_to_annotation_file(image_file_path):
    path_with_out_last_folder=image_file_path.split(os.path.sep)[:-2]
    return os.path.join(*path_with_out_last_folder,'annotations',os.path.basename(image_file_path).replace('.jpg','.json'))

def convert_name_yolo_annotation_file_to_image_file(annotation_file_path):
    path_with_out_last_folder=annotation_file_path.split(os.path.sep)[:-2]
    return os.path.join(*path_with_out_last_folder,'images',os.path.basename(annotation_file_path).replace('.txt','.jpg'))

def convert_name_image_file_to_yolo_annotation_file(image_file_path):
    path_with_out_last_folder=image_file_path.split(os.path.sep)[:-2]
    return os.path.join(*path_with_out_last_folder,'annotations',os.path.basename(image_file_path).replace('.jpg','.txt'))

def draw_bound_box_diopsis(image_file,destination_path):
    image=cv2.imread(image_file)
    img_w, img_h = image.shape[1],image.shape[0]
    annotation_file=convert_name_image_file_to_annotation_file(image_file)
    list_box=[]
    annotation_json=json.load(open(annotation_file))
    insect_count=0
    for annotation in annotation_json['annotations']:
        x = annotation['shape']['x']
        y = annotation['shape']['y']
        w = annotation['shape']['width']
        h = annotation['shape']['height']
        cv2.rectangle(image, (int(x),int(y)), 
                                  (int(x+w),int(y+h)), 
                                   (0, 255, 0), 2)
        insect_count += 1
    save_image_file=f"{destination_path}/{insect_count}_{os.path.basename(image_file)}"
    cv2.imwrite(save_image_file,image)  

def draw_bound_box_preds_targets_metrics(image_file,destination_path,preds,targets,extra_str=None,ext_str_file=None):
    image=cv2.imread(image_file)
    img_w, img_h = image.shape[1],image.shape[0]
    count_ped=0
    count_target=0
    for pred in preds:
        for box in zip(pred['boxes'],pred['labels'],pred['scores']):
            x1 = box[0][0].cpu().detach().item()
            y1 = box[0][1].cpu().detach().item()
            x2 = box[0][2].cpu().detach().item()
            y2 = box[0][3].cpu().detach().item()
            count_ped +=1
            cv2.putText(image, 
                        f"{round(box[2].cpu().detach().item(),2)} ", 
                        (int(x1),int(y1)-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0),
                        1
                        )
            cv2.rectangle(image, (int(x1),int(y1)), 
                                      (int(x2),int(y2)), 
                                       (0, 255, 0), 2)
    for target in targets:
        for box in zip(target['boxes'],target['labels']):
            x1 = box[0][0].cpu().detach().item()
            y1 = box[0][1].cpu().detach().item()
            x2 = box[0][2].cpu().detach().item()
            y2 = box[0][3].cpu().detach().item()
            count_target +=1
            cv2.rectangle(image, (int(x1),int(y1)),
                                      (int(x2),int(y2)),
                                       (255, 0, 0), 2)        
    font = cv2.FONT_HERSHEY_SIMPLEX
    if extra_str is None:     
        image = cv2.putText(image, f"P:{count_ped}/T:{count_target}", (50,50), font, 
                   1, (255,255,255), 2, cv2.LINE_AA)
    else:
        image = cv2.putText(image, f"P:{count_ped}/T:{count_target} {extra_str}", (50,50), font, 
                   1, (255,255,255), 2, cv2.LINE_AA)
 
    if ext_str_file is not None:
        save_image_file=f"{destination_path}/{os.path.basename(image_file)}"
    else:
        save_image_file=f"{destination_path}/{ext_str_file}_{os.path.basename(image_file)}"
    cv2.imwrite(save_image_file,image)


def draw_bound_box_yolo_format(yolo_image,yolo_annotation,yolo_folder,image_size=None):
    image=cv2.imread(yolo_image)
    if image_size is not None:
        image= cv2.resize(image, image_size)
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
    create_folders(yolo_annotated_image_folder)
    yolo_annotated_image=os.path.join(yolo_annotated_image_folder, os.path.basename(yolo_image))
    
    cv2.imwrite(yolo_annotated_image,image)   

def copy_symlink(source_link, destination_link):
    
    if not os.path.islink(source_link):
        os.symlink(source_link, destination_link)
    else:
    
        original_target = os.readlink(source_link)

        
        try:
            os.symlink(original_target, destination_link)
            print(f"Symbolic link copied: {destination_link} -> {original_target}")
        except FileExistsError:
        
            os.remove(destination_link)
            os.symlink(original_target, destination_link)
            print(f"Existing link replaced: {destination_link} -> {original_target}")
        except OSError as e:
            print(f"Error creating symbolic link: {e}")

def copy_links(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, os.path.relpath(source_path, source_folder))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            if os.path.islink(destination_path):
                os.unlink(destination_path)
            copy_symlink(os.path.abspath(source_path),os.path.abspath(destination_path))

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    :param box1: A list of four coordinates (x1, y1, x2, y2) of the first box
    :param box2: A list of four coordinates (x1, y1, x2, y2) of the second box
    :return: The IoU value as a float between 0 and 1
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def yolo_to_xyxy_tensor(boxes, img_width, img_height):
    """
    Μετατρέπει bounding boxes από μορφή YOLO (cx, cy, w, h) σε μορφή (x1, y1, x2, y2).
    
    :param boxes: Tensor μορφής (..., 4) που περιέχει τα bounding boxes σε μορφή YOLO (cx, cy, w, h)
    :param img_size: Tensor με δύο στοιχεία [width, height] της εικόνας
    :return: Tensor με τα bounding boxes σε μορφή (x1, y1, x2, y2)
    """
    # Διασφάλιση ότι το img_size είναι tensor
    
    # Αποκανονικοποίηση
    boxes = boxes * torch.tensor((img_width, img_height, img_width, img_height)).to(boxes.device)
    
    # Υπολογισμός των x1, y1, x2, y2
    x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
    x2y2 = boxes[..., :2] + boxes[..., 2:] / 2
    
    return torch.cat([x1y1, x2y2], dim=-1)
def xyxy_to_yolo(boxes, img_width, img_height):
    """
    Μετατρέπει bounding boxes από μορφή (x1, y1, x2, y2) σε μορφή YOLO (cx, cy, w, h).
    
    :param boxes: Tensor μορφής (..., 4) που περιέχει τα bounding boxes σε μορφή (x1, y1, x2, y2)
    :param img_width: Πλάτος της εικόνας
    :param img_height: Ύψος της εικόνας
    :return: Tensor με τα bounding boxes σε μορφή YOLO (cx, cy, w, h)
    """

    # Αντιγράφουμε το tensor για να μην τροποποιήσουμε το αρχικό
    yolo_boxes = torch.tensor(boxes) 
    
    # Υπολογισμός πλάτους και ύψους των boxes
    yolo_boxes[..., 2] = boxes[..., 2] - boxes[..., 0]  # πλάτος
    yolo_boxes[..., 3] = boxes[..., 3] - boxes[..., 1]  # ύψος
    
    # Υπολογισμός κεντρικών σημείων
    yolo_boxes[..., 0] = boxes[..., 0] + yolo_boxes[..., 2] / 2  # cx
    yolo_boxes[..., 1] = boxes[..., 1] + yolo_boxes[..., 3] / 2  # cy
    
    # Κανονικοποίηση με βάση τις διαστάσεις της εικόνας
    yolo_boxes[..., 0] /= img_width
    yolo_boxes[..., 1] /= img_height
    yolo_boxes[..., 2] /= img_width
    yolo_boxes[..., 3] /= img_height

    return yolo_boxes
  

