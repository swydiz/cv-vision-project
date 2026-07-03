from ultralytics import YOLO


def train_yolov5():
    # ✔ загружаем YOLOv5s веса
    model = YOLO("yolov5s.pt")

    # ✔ обучение
    results = model.train(
        data="configs/coco.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,          # GPU (или "cpu")
        workers=8,
        project="runs/train",
        name="yolov5_experiment",
        patience=10
    )

    print("Training finished!")
    return results

def test_model():
    model = YOLO("runs/detect/runs/train/yolov5_experiment-6/weights/best.pt")

    results = model.predict(
        source="data_small/images/val",
        conf=0.25,
        save=True,
        imgsz=640
    )

    print("Inference done!")


if __name__ == "__main__":
    test_model()
    #train_yolov5()