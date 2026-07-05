import os
import glob
import cv2
import torch

def set_seed(seed=42):
    import random, numpy, torch
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_class_name(label, classes):
    if label is None:
        return "объект"

    label = int(label)

    if 0 <= label < len(classes):
        return classes[label]

    return f"объект {label}"

def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    return image


def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def get_image_size(image):
    return image.shape[1], image.shape[0]

def get_image_paths(folder, num_images=10):
    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )
    return paths[:num_images]


def load_image_cv(path):
    return cv2.imread(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image(path, image):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)


def filter_predictions(boxes, labels, scores, threshold=0.5):
    keep = scores >= threshold
    return boxes[keep], labels[keep], scores[keep]


def draw_boxes(image, boxes, labels, scores, conf=0.5):
    for box, label, score in zip(boxes, labels, scores):

        if score < conf:
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = get_class_name(label)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            f"{class_name}: {score:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    return image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)