import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple

from configs.config import DEVICE, COCO_CLASSES
from src.dataset.dataset import YOLODataset


def load_ground_truth(image_path: str, label_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает ground truth для одного изображения.
    
    Args:
        image_path: путь к изображению
        label_dir: директория с YOLO-разметкой
        
    Returns:
        boxes: массив [N, 4] в формате [x1, y1, x2, y2]
        labels: массив [N] с классами (0-based)
    """
    img_name = os.path.basename(image_path)
    label_path = os.path.join(
        label_dir,
        img_name.rsplit(".", 1)[0] + ".txt"
    )
    
    # Получаем размеры изображения
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    boxes = []
    labels = []
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                cls = int(cls)
                
                xc *= w
                yc *= h
                bw *= w
                bh *= h
                
                x1 = xc - bw / 2
                y1 = yc - bh / 2
                x2 = xc + bw / 2
                y2 = yc + bh / 2
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
    
    return np.array(boxes), np.array(labels)


def collect_predictions_faster_rcnn(
    model,
    image_paths: List[str],
    label_dir: str,
    confidence: float = 0.3,
    device=DEVICE
) -> List[Dict]:
    """
    Собирает предсказания для Faster R-CNN.
    
    Returns:
        Список словарей с предсказаниями и ground truth
    """
    results = []
    model.eval()
    
    for image_path in image_paths:
        # Загружаем изображение
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Преобразуем в тензор
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)
        
        # Предсказание
        with torch.no_grad():
            prediction = model([tensor])[0]
        
        # Фильтруем по confidence
        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        
        # Применяем порог
        keep = scores >= confidence
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # Важно: для Faster R-CNN метки сдвинуты на +1 (0 - фон)
        # Поэтому вычитаем 1, чтобы получить COCO-индексы
        labels = labels - 1
        
        # Загружаем ground truth
        gt_boxes, gt_labels = load_ground_truth(image_path, label_dir)
        
        results.append({
            "pred_boxes": boxes,
            "pred_labels": labels,
            "pred_scores": scores,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "image_path": image_path
        })
    
    return results


def collect_predictions_ssd(
    model,
    image_paths: List[str],
    label_dir: str,
    confidence: float = 0.3,
    device=DEVICE
) -> List[Dict]:
    """
    Собирает предсказания для SSD.
    """
    results = []
    model.eval()
    
    for image_path in image_paths:
        # Загружаем изображение
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ресайзим до 512x512 (как при обучении)
        image_resized = cv2.resize(image_rgb, (512, 512))
        h, w = image.shape[:2]
        
        # Преобразуем в тензор
        tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)
        
        # Предсказание
        with torch.no_grad():
            prediction = model([tensor])[0]
        
        # Фильтруем по confidence
        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        
        keep = scores >= confidence
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # SSD также использует сдвиг на +1
        labels = labels - 1
        
        # Масштабируем координаты обратно к оригинальному размеру
        scale_x = w / 512
        scale_y = h / 512
        boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
        
        # Загружаем ground truth
        gt_boxes, gt_labels = load_ground_truth(image_path, label_dir)
        
        results.append({
            "pred_boxes": boxes,
            "pred_labels": labels,
            "pred_scores": scores,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "image_path": image_path
        })
    
    return results


def collect_predictions_yolo(
    model,
    image_paths: List[str],
    label_dir: str,
    confidence: float = 0.25,
) -> List[Dict]:
    """
    Собирает предсказания для YOLO моделей.
    """
    results = []
    
    for image_path in image_paths:
        # Предсказание YOLO
        results_yolo = model.predict(
            source=image_path,
            conf=confidence,
            verbose=False
        )[0]
        
        # Получаем размеры
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        boxes = []
        labels = []
        scores = []
        
        if results_yolo.boxes is not None:
            boxes = results_yolo.boxes.xyxy.cpu().numpy()
            labels = results_yolo.boxes.cls.cpu().numpy().astype(int)
            scores = results_yolo.boxes.conf.cpu().numpy()
        
        # Загружаем ground truth
        gt_boxes, gt_labels = load_ground_truth(image_path, label_dir)
        
        results.append({
            "pred_boxes": boxes,
            "pred_labels": labels,
            "pred_scores": scores,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "image_path": image_path
        })
    
    return results