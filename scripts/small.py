import os
import random
import shutil
from pathlib import Path


def reduce_split(img_dir, lbl_dir, out_dir, split_name, size, seed=42):
    random.seed(seed)

    images = [
        f for f in os.listdir(img_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    random.shuffle(images)
    images = images[:size]

    out_img = Path(out_dir) / "images" / split_name
    out_lbl = Path(out_dir) / "labels" / split_name

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for img_name in images:
        # image copy
        shutil.copy(
            os.path.join(img_dir, img_name),
            out_img / img_name
        )

        # label copy
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_name)

        if os.path.exists(label_path):
            shutil.copy(label_path, out_lbl / label_name)

    print(f"{split_name}: {len(images)}")


def main():
    base = "data/raw"

    reduce_split(
        img_dir=f"{base}/images/train2017",
        lbl_dir=f"{base}/labels/train2017",
        out_dir="data_small",
        split_name="train",
        size=5000
    )

    reduce_split(
        img_dir=f"{base}/images/val2017",
        lbl_dir=f"{base}/labels/val2017",
        out_dir="data_small",
        split_name="val",
        size=2000
    )


if __name__ == "__main__":
    main()