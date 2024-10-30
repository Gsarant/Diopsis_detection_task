import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import ConfusionMatrix
import torchmetrics
from torchvision.ops import box_iou


class Object_detection_metrics:
    def __init__(self,preds,targets,device='cpu'):
        # Mean Average Precision
        self.preds=preds
        self.targets=targets
        self.map_metric = MeanAveragePrecision(box_format='xyxy',
                                                iou_type='bbox',
                                                )
        self.map_metric=self.map_metric.to(device)
    def map_calculate(self):    
        MAP_results=[]
        self.map_metric.update(self.preds, self.targets)
        map_results = self.map_metric.compute()
        print("Mean Average Precision Results:")
        for k, v in map_results.items():
            MAP_results.append(f"{k}: {v}")
           # print(f"{k}: {v}")
        return MAP_results
    def confusion_matrix(self):
        pred_labels = torch.cat([self.pred['labels'] for self.pred in self.preds])
        target_labels = torch.cat([self.target['labels'] for self.target in self.targets])
        try:
            num_classes = max(pred_labels.max(), target_labels.max()) + 1
        except:
            print("Confusion Matrix not available for binary classification")
            return  None

        confusion_matrix = ConfusionMatrix(task="binary",num_classes=num_classes)
        confusion_matrix_result = confusion_matrix(pred_labels, target_labels)
        print("\nConfusion Matrix:")
        print(confusion_matrix_result)
        return confusion_matrix_result
          
            
    def IoU(self):
        # IoU (Intersection over Union)
        # torchmetrics doesn't have a built-in IoU metric for object detection
        # but we can use the one from torchvision
        pred_boxes = self.preds[0]['boxes']
        target_boxes = self.targets[0]['boxes']
        iou = box_iou(pred_boxes, target_boxes)
        print("\nIoU:")
        print(iou)
        return iou
    def Precision(self):
        # Precision, Recall, F1 Score
        precision = torchmetrics.Precision(task='binary')
        #binary_preds = (self.preds[0]['scores'] > 0.5).int()
        binary_preds = self.preds[0]['labels']
        precision_result = precision(binary_preds, self.targets[0]['labels'])
        print(f"\nPrecision: {precision_result}")
        return precision_result

    def Recall(self):
        recall = torchmetrics.Recall(task='binary')
        #binary_preds = (self.preds[0]['scores'] > 0.5).int()
        binary_preds = self.preds[0]['labels']
        recall_result = recall(binary_preds, self.targets[0]['labels'])
        print(f"Recall: {recall_result}")
        return recall_result
    
    def F1(self):
        f1_score = torchmetrics.F1Score(task='binary')
        # Assuming scores above 0.5 are positive predictions
        #binary_preds = (self.preds[0]['scores'] > 0.5).int()
        binary_preds = self.preds[0]['labels']
        f1_result = f1_score(binary_preds, self.targets[0]['labels'])
        print(f"F1 Score: {f1_result}")
        return f1_result
