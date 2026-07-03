import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.dataset.dataset import YOLODataset, collate_fn
from src.models.efficient import get_model
from configs.config import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

def get_predict_model(num_classes):
    config = get_efficientdet_config("tf_efficientdet_d0")
    config.num_classes = num_classes
    config.image_size = (512, 512)

    net = EfficientDet(config, pretrained_backbone=False)

    model = DetBenchPredict(net)
    return model


def draw(image, boxes, scores, labels, score_thr=0.3):
    image = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    h, w, _ = image.shape

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box

        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
        )

        plt.text(
            x1,
            y1,
            f"{int(label)}:{score:.2f}",
            color="white",
            bbox=dict(facecolor="red", alpha=0.5)
        )

    plt.axis("off")
    plt.show()


def test():
    print("Device:", DEVICE)

    # -----------------------
    # Dataset
    # -----------------------
    dataset = YOLODataset(
        TRAIN_IMAGES,
        TRAIN_LABELS,
        max_images=20
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # -----------------------
    # Model
    # -----------------------
    model = get_predict_model(NUM_CLASSES)
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_effdet.pth"), map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    print("Testing started...")

    score_thr = 0.2

    with torch.no_grad():
        for images, targets in loader:

            images = torch.stack(images).to(DEVICE)

            outputs = model(images)

            for i, pred in enumerate(outputs):

                pred = pred.detach().cpu()

                # -----------------------
                # CASE: tensor output [N,6]
                # -----------------------
                if isinstance(pred, torch.Tensor):

                    if pred.ndim != 2 or pred.shape[1] < 6:
                        print("Bad output shape:", pred.shape)
                        continue

                    boxes = pred[:, 0:4]
                    scores = pred[:, 4]
                    labels = pred[:, 5].long()

                else:
                    print("Unexpected output type:", type(pred))
                    continue

                # -----------------------
                # FILTER LOW CONFIDENCE
                # -----------------------
                mask = scores > score_thr

                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                print(f"\nPredictions after filter: {len(boxes)}")

                # -----------------------
                # DEBUG GT (очень полезно)
                # -----------------------
                gt = targets[i]
                print("GT boxes:", gt["boxes"].shape)

                # -----------------------
                # VISUALIZATION
                # -----------------------
                draw(images[i], boxes, scores, labels, score_thr)


    print("DONE")


if __name__ == "__main__":
    test()