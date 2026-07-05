import os
import cv2
import torch
import argparse
import glob
import pyttsx3

from ultralytics import YOLO

from configs.config import DEVICE, COCO_CLASSES_RU
from src.models.faster_rcnn import get_model as get_faster
from src.models.ssd import get_model as get_ssd
from src.inference.scene_description import get_position, build_description
from src.utils.utils import get_class_name

engine = pyttsx3.init()
engine.setProperty("rate", 170)


def set_russian_voice(engine):
    voices = engine.getProperty("voices")
    for v in voices:
        if "russian" in v.name.lower() or "ru" in v.id.lower():
            engine.setProperty("voice", v.id)
            return


set_russian_voice(engine)


def speak(text, enable=True):
    if enable:
        engine.say(text)
        engine.runAndWait()


def load_model(name: str):

    if name == "yolov8n":
        return YOLO("results/models/best_yolov8n.pt")

    elif name == "yolov8s":
        return YOLO("results/models/best_yolov8s.pt")

    elif name == "yolov5":
        return YOLO("results/models/best_yolov5.pt")

    elif name == "ssd":
        model = get_ssd(91)
        model.load_state_dict(torch.load("results/models/best_ssd.pth", map_location=DEVICE))
        model.to(DEVICE).eval()
        return model

    elif name == "faster":
        model = get_faster(81)
        model.load_state_dict(torch.load("results/models/best_faster.pth", map_location=DEVICE))
        model.to(DEVICE).eval()
        return model

    else:
        raise ValueError("Unknown model")


def predict(model, model_name, image_path, speak_text=True, save=True):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    h, w = image.shape[:2]

    objects = []  
    annotated = image.copy()

    if model_name in ["yolov8n", "yolov8s", "yolov5"]:

        res = model.predict(image_path, conf=0.25, verbose=False)[0]

        if res.boxes is not None:

            boxes = res.boxes.xyxy.cpu().numpy()
            labels = res.boxes.cls.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):

                if score < 0.25:
                    continue

                x1, y1, x2, y2 = map(int, box)

                name = COCO_CLASSES_RU[int(label)]
                pos = get_position(x1, y1, x2, y2, w, h)

                objects.append((name, pos, (x1, y1, x2, y2)))

    else:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            pred = model([tensor])[0]

        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):

            if score < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box)

            label = int(label)

            if 1 <= label <= len(COCO_CLASSES_RU):
                name = COCO_CLASSES_RU[label - 1]
            else:
                name = "объект"

            pos = get_position(x1, y1, x2, y2, w, h)

            objects.append((name, pos, (x1, y1, x2, y2)))

    sentences = []

    for name, pos, box in objects:

        x1, y1, x2, y2 = box

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            annotated,
            name,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        sentences.append(f"Я вижу {name} {pos}")

    description = " ".join(sentences)

    print(description)

    if speak_text:
        speak(description)

    if save:
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", "result.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")

    return annotated, description


def run_folder(model, model_name, folder):

    images = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:10]

    print(f"Found {len(images)} images")

    for img in images:
        print("\nProcessing:", img)
        predict(model, model_name, img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n")
    parser.add_argument("--image", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--no_speak", action="store_true")

    args = parser.parse_args()

    model = load_model(args.model)

    if args.folder:
        run_folder(model, args.model, args.folder)

    elif args.image:
        predict(model, args.model, args.image, speak_text=not args.no_speak)

    else:
        print("Use --image or --folder")