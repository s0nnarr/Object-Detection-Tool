from ultralytics import YOLO

def main():
    model = YOLO("sku_baseline.pt")
    model.train(
        data="sku_local.yaml",
        epochs=8,
        imgsz=640,
        batch=8,
        workers=4
    )
if __name__ == "__main__":
    main()