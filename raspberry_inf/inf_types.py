import cv2
import supervision as sv
import torch
import numpy as np
import torchvision.ops as ops
from copy import deepcopy

class YOLOv8Ensemble:
    def __init__(self, models_info, weights=None,device='cpu'):
        self.models_info = models_info
        self.weights = weights if weights else [1/len(models_info)] * len(models_info)
        self.device = device
    
    def predict(self, image_file_name, conf=0.25, iou=0.45):
        all_predictions = []
        for model_info, weight in zip( self.models_info, self.weights):

            results,original_image=make_pedicts(model=model_info['model'],
                                               image_file_name=image_file_name,
                                               device=self.device,
                                               conf=conf,
                                               image_size=model_info['image_size'],
                                               method=model_info['method']
                                                )
            if  model_info['method'] == 'default':
                predictions=results[0].boxes
                combined = torch.cat((
                        predictions.xyxy,
                        predictions.conf.unsqueeze(-1),
                        predictions.cls.unsqueeze(-1)
                ), dim=-1).to( self.device)
                all_predictions.append(combined )
        
            else:
                results_confidence=[]
                results_xyxy=[]
                results_class_id=[]
                for index in range(len(results.confidence)):
                        results_confidence.append(results.confidence[index])
                        results_xyxy.append(results.xyxy[index])
                        results_class_id.append(results.class_id[index])
                if len(results_xyxy) > 0:
                    combined = torch.cat((
                            torch.tensor(results_xyxy).to( self.device),
                            torch.tensor(results_confidence).to( self.device).unsqueeze(-1),
                            torch.tensor(results_class_id).to( self.device).unsqueeze(-1)
                    ), dim=-1)
                    all_predictions.append(combined )
                

        all_predictions = torch.cat(all_predictions, dim=0)

        keep = ops.nms(
            boxes=all_predictions[:, :4],
            scores=all_predictions[:, 4],
            iou_threshold=iou
        )
        
        nms_results = all_predictions[keep]
        
        # Φιλτράρισμα με βάση το confidence threshold
        conf_mask = nms_results[:, 4] >= conf
        nms_results = nms_results[conf_mask]
        weighted_results = self.apply_weights(nms_results)

        return weighted_results
    
    def apply_weights(self, results):
        # Υπολογισμός του weighted average για κάθε bounding box
        weighted_results = []
        if len(results) == 0:
            return results
        for box in results:
            weighted_box = box.clone()
            for i, model_info in enumerate(self.models_info):
                if box[5] in model_info['model'].names.values():  # Ελέγχουμε αν η κλάση υπάρχει στο μοντέλο
                    weighted_box[4] *= self.weights[i]
            weighted_results.append(weighted_box)
        return torch.stack(weighted_results)
    


def make_pedicts(model,image_file_name,device,conf=0.5,image_size=[640,640],method='default'):
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)
    
    original_image=cv2.imread(image_file_name)
    
    if method == 'default' or method == 'average':
      
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #image=cv2.resize(original_image,(640,640))
        results=model.predict(image_file_name,
                              conf=conf,
                              imgsz=image_size,
                              iou=0.45
                              )
                
        return results,original_image
    else:
        slicer = sv.InferenceSlicer(callback = callback,
                                    slice_wh=image_size,
                                    overlap_ratio_wh=(0.0, 0.0),
                                    iou_threshold=0.6,
                                    thread_workers=1,
                                    )
        results = slicer(original_image)
        results['boxes']=np.array(results['xyxy'])
        return results,original_image
    
def average_yolov8_models( models, weights,device,output_path=None):
    
    if len(models) != len(weights):
        raise ValueError("Ο αριθμός των μοντέλων πρέπει να είναι ίσος με τον αριθμό των βαρών.")
    
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Τα βάρη πρέπει να αθροίζονται στο 1.")


    if not all(m.model.yaml == models[0].model.yaml for m in models):
        raise ValueError("Όλα τα μοντέλα πρέπει να έχουν την ίδια αρχιτεκτονική.")

    
    averaged_model = deepcopy(models[0])
    
    state_dict = averaged_model.model.state_dict()
    for key in state_dict.keys():
        if state_dict[key].dtype in [torch.float32, torch.float16]:
            # Υπολογισμός του μέσου όρου των βαρών
            #mean_weight = torch.mean(torch.stack([m.model.state_dict()[key].to(state_dict[key].dtype).to(state_dict[key].device) for m in models]), dim=0)
            #state_dict[key] = mean_weight
            weighted_sum = torch.zeros_like(state_dict[key])
            for model, weight in zip(models, weights):
                weighted_sum += weight * model.model.state_dict()[key].to(state_dict[key].dtype).to(state_dict[key].device)
            state_dict[key] = weighted_sum

    # Φόρτωση των μέσων βαρών πίσω στο μοντέλο
    averaged_model.model.load_state_dict(state_dict)
    
    # Ενημέρωση των μεταδεδομένων του μοντέλου
    averaged_model.model.names = models[0].model.names
    averaged_model.model.yaml = models[0].model.yaml
    
    #averaged_model.save(output_path)
    #print(f"Το συνδυασμένο μοντέλο αποθηκεύτηκε στο {output_path}")
    
    return averaged_model