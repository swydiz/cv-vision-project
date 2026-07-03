"""
Обучение YOLO модели.
"""

from ultralytics import YOLO
import yaml

def load_config():
    """Загружает configs/default.yaml"""
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_yolo(model_name='yolov8n.pt'):
    # Загружаем конфиг
    params = load_config()
    
    # Загружаем модель
    model = YOLO(model_name)

    # Запускаем обучение
    model.train(**params)  

if __name__ == "__main__":
    train_yolo()