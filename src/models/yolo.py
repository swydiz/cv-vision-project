from ultralytics import YOLO
import torch.nn as nn
import cv2
import numpy as np


class YOLOModel(nn.Module):  
    def __init__(self, model_name='yolov8s.pt', num_classes=80):
        super().__init__()
        self.model = YOLO(model_name)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, image, conf=0.5, iou=0.5):
        results = self.model(image, conf=conf, iou=iou)
        
        boxes, labels, scores = [], [], []
        if results and len(results) > 0:
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        labels.append(int(box.cls[0]))
                        scores.append(float(box.conf[0]))
        return boxes, labels, scores
    
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640, **kwargs):
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            **kwargs
        )
        return results
    
    def load_pretrained(self, path):
        self.model = YOLO(path)
    
    def save(self, path):
        self.model.save(path)