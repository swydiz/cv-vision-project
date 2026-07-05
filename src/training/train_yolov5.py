from ultralytics import YOLO


def train_yolov5():
    model = YOLO("yolov5s.pt")

    results = model.train(
        data="configs/coco.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,          
        workers=8,
        project="runs/train",
        name="yolov5_experiment",
        patience=10
    )

    print("Training finished!")
    return results


if __name__ == "__main__":
    train_yolov5()