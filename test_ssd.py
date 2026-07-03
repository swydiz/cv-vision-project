import os
import cv2
import torch
import torchvision

from configs.config import *
from src.models.ssd import get_model


CONFIDENCE_THRESHOLD = 0.2


def predict(image_path):

    device = DEVICE

    model = get_model(NUM_CLASSES)

    state = torch.load(
        os.path.join(SAVE_DIR, "best_ssd.pth"),
        map_location=device
    )

    result = model.load_state_dict(state)

    print(result)

    model.to(device)
    model.eval()

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original = image.copy()

    image = cv2.resize(image, (512, 512))

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float() / 255.0
    )

    tensor = tensor.to(device)

    with torch.no_grad():
        prediction = model([tensor])[0]
        print("Scores:")
        print(prediction["scores"][:20])

        print("Labels:")
        print(prediction["labels"][:20])

        print("Boxes:", len(prediction["boxes"]))

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):

        if score < CONFIDENCE_THRESHOLD:
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
            f"{label}: {score:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    os.makedirs("results", exist_ok=True)

    save_path = "results/ssd_result.jpg"

    cv2.imwrite(save_path, image)

    print(f"Result saved: {save_path}")

if __name__ == "__main__":

    predict("data/raw/images/val2017/000000000139.jpg")