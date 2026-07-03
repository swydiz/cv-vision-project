import torch

# ------------------------
# Пути
# ------------------------

TRAIN_IMAGES = "data/raw/images/train2017"
TRAIN_LABELS = "data/raw/labels/train2017"

VAL_IMAGES = "data/raw/images/val2017"
VAL_LABELS = "data/raw/labels/val2017"

SAVE_DIR = "results/models"

# ------------------------
# Датасет
# ------------------------

NUM_CLASSES = 80      # COCO

MAX_TRAIN_IMAGES = 5000
MAX_VAL_IMAGES = 2000

# ------------------------
# Обучение
# ------------------------

BATCH_SIZE = 8

EPOCHS = 30

LEARNING_RATE = 0.005

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005

NUM_WORKERS = 2


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)