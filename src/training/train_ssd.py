import os
import time
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from PIL import Image

import torchvision
from torchvision.transforms import v2 as T

from configs.config import *
from src.models.ssd import get_model


def get_transform(train=True):
    transforms = []

    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


class DetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, max_images=None, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms

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

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        label_path = os.path.join(
            self.label_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
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

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

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

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train():

    device = DEVICE
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Device:", device)

    train_dataset = DetectionDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        max_images=MAX_TRAIN_IMAGES,
        transforms=get_transform(train=True)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = get_model(NUM_CLASSES).to(device)

    optimizer = SGD(
        model.parameters(),
        lr=0.002,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )

    start = time.time()
    best_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        epoch_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, targets in progress:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        lr_scheduler.step()

        epoch_loss /= len(train_loader)

        print(f"\nEpoch {epoch+1} | Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_ssd.pth"))

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ssd_epoch_{epoch+1}.pth"))

    print("\nDONE")
    print("Time:", (time.time() - start) / 60, "min")
    print("Best loss:", best_loss)


if __name__ == "__main__":
    train()