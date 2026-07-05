import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, max_images=None, seed=42):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

        if max_images is not None:
            random.seed(seed)
            self.images = random.sample(self.images, max_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(
            self.label_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        img = Image.open(img_path).convert("RGB")

        img = np.array(img)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        h, w = img.shape[1], img.shape[2]

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
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

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))