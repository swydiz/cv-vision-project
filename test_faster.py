import cv2
import torch
import torchvision
from torchvision.transforms import v2 as T

from configs.config import DEVICE, NUM_CLASSES
from src.models.faster_rcnn import get_model


# ------------------------
# ПУТИ
# ------------------------

MODEL_PATH = "results/models/best.pth"
IMAGE_PATH = "data/raw/images/val2017/000000000632.jpg"   # своё изображение
OUTPUT_PATH = "results/test_result.jpg"

# ------------------------
# ЗАГРУЗКА МОДЕЛИ
# ------------------------

model = get_model(NUM_CLASSES)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# ------------------------
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# ------------------------

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

tensor = transform(image).to(DEVICE)

# ------------------------
# ПРЕДСКАЗАНИЕ
# ------------------------

with torch.no_grad():
    prediction = model([tensor])[0]

boxes = prediction["boxes"].cpu().numpy()
scores = prediction["scores"].cpu().numpy()
labels = prediction["labels"].cpu().numpy()

# ------------------------
# ОТОБРАЖЕНИЕ
# ------------------------

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

CONFIDENCE = 0.5

for box, score, label in zip(boxes, scores, labels):

    if score < CONFIDENCE:
        continue

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"{label}:{score:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

cv2.imwrite(OUTPUT_PATH, image)

print("Результат сохранён:", OUTPUT_PATH)

