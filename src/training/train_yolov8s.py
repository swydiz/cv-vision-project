from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO('yolov8s.pt')

    model.train(
        data='configs/coco.yaml',
        epochs=30,
        batch=12,
        imgsz=640,
        device=0,
        plots=True,
        save=True,
        project='runs/train/',
        name='yolov8s_30_epochs',
        exist_ok=True
    )

    print("YOLOv8s обучена!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()