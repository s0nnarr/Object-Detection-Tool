from ultralytics import YOLO

def main():
    model = YOLO("yolo26m.pt")
    model.train(
        data="E:/yolo_dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=4
    )
if __name__ == "__main__":
    main()