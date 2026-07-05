import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import DEVICE, NUM_CLASSES
from src.models.ssd import get_model
from src.utils.utils import get_image_paths
from src.evaluation.metrics import evaluate_dataset, save_metrics
from src.evaluation.collect_predictions import collect_predictions_ssd

MODEL_PATH = "results/models/best_ssd.pth"
IMAGE_DIR = "data/raw/val2017"
LABEL_DIR = "data/processed/yolo_labels/val2017"
OUTPUT_DIR = "results/metrics"

CONFIDENCE = 0.2
NUM_IMAGES = 50
IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("Оценка модели SSD")
print("=" * 50)

print(f"Загрузка модели: {MODEL_PATH}")
model = get_model(91)  # +1 для фона
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

image_paths = get_image_paths(IMAGE_DIR, NUM_IMAGES)
print(f"Найдено {len(image_paths)} изображений")

print("Выполнение инференса...")
results = collect_predictions_ssd(
    model=model,
    image_paths=image_paths,
    label_dir=LABEL_DIR,
    confidence=CONFIDENCE,
    device=DEVICE
)

print("Расчет метрик...")
metrics = evaluate_dataset(results)

metrics["Model"] = "SSD"
metrics["Confidence"] = CONFIDENCE
metrics["IOU_Threshold"] = IOU_THRESHOLD
metrics["Num_Images"] = NUM_IMAGES

print("\nРезультаты оценки:")
print(f"  Precision: {metrics['Precision']:.4f}")
print(f"  Recall:    {metrics['Recall']:.4f}")
print(f"  F1-score:  {metrics['F1-score']:.4f}")
print(f"  Mean IoU:  {metrics['Mean IoU']:.4f}")
print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

save_path = os.path.join(OUTPUT_DIR, "ssd_metrics.csv")
save_metrics(metrics, save_path)

print(f"\nГотово! Результаты сохранены в {save_path}")