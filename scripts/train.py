from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(
    # experiment with hyperparameters
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)

