import torch

# ------------------------
# Пути
# ------------------------

TRAIN_IMAGES = "data/raw/train2017"
TRAIN_LABELS = "data/raw/train2017"

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

MODEL_CLASS_OFFSET = {
    "yolov8n": 0,
    "yolov8s": 0,
    "yolov5": 0,
    "ssd": 1,
    "faster": 1
}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

COCO_CLASSES_RU = [
    "человек", "велосипед", "машина", "мотоцикл", "самолет",
    "автобус", "поезд", "грузовик", "лодка", "светофор",
    "пожарный гидрант", "стоп-знак", "парковочный счетчик", "скамейка",
    "птица", "кошка", "собака", "лошадь", "овца",
    "корова", "слон", "медведь", "зебра", "жираф",
    "рюкзак", "зонт", "сумка", "галстук", "чемодан",
    "фрисби", "лыжи", "сноуборд", "мяч", "змей",
    "бита", "перчатка", "скейтборд", "доска для серфинга", "ракетка",
    "бутылка", "бокал", "чашка", "вилка", "нож",
    "ложка", "миска", "банан", "яблоко", "сэндвич",
    "апельсин", "брокколи", "морковь", "хот-дог", "пицца",
    "пончик", "торт", "стул", "диван", "горшок для растений",
    "кровать", "стол", "туалет", "телевизор", "ноутбук",
    "мышь", "пульт", "клавиатура", "телефон", "микроволновка",
    "духовка", "тостер", "раковина", "холодильник", "книга",
    "часы", "ваза", "ножницы", "плюшевый мишка", "фен", "зубная щетка"
]