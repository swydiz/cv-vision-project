import os
import time
import torch
import csv
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from configs.config import *
from src.models.faster_rcnn import get_model

torch.backends.cudnn.benchmark = True

class FastDataset(Dataset):
    def __init__(self, image_dir, label_dir, max_images=None):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = [
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        if max_images:
            self.images = self.images[:max_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (512, 512))

        h, w = 512, 512

        label_path = os.path.join(
            self.label_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())

                    cls = int(cls) + 1 

                    xc *= w
                    yc *= h
                    bw *= w
                    bh *= h

                    x1 = xc - bw / 2
                    y1 = yc - bh / 2
                    x2 = xc + bw / 2
                    y2 = yc + bh / 2

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if len(boxes) > 0 else torch.tensor([0.0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train():

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Device:", DEVICE)

    train_dataset = FastDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        max_images=MAX_TRAIN_IMAGES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,       
        pin_memory=False,
        collate_fn=collate_fn
    )

    model = get_model(NUM_CLASSES).to(DEVICE)

    optimizer = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler("cuda")

    history = []
    best_loss = float("inf")

    start = time.time()

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for images, targets in progress:

            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # 🔥 AMP (only here, safe)
            with autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

    print("\nDONE")
    print("Time:", (time.time() - start) / 60, "min")


if __name__ == "__main__":
    train()