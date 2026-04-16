from ultralytics import YOLO
import os
from roboflow import Roboflow
from huggingface_hub import hf_hub_download



# Download the Huggingface model


def main():
    # model = YOLO("sku_baseline.pt")
    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    rf_workspace = rf.workspace("stancus-workspace").project("my-first-project-uj5as")
    dataset = rf_workspace.version(1).download("yolov8")

    model_path = hf_hub_download(
        repo_id="s0narr/aisight-model",
        filename="best.pt",
        token=os.environ.get("HF_ACCESS_TOKEN")
    )

    model = YOLO(model_path)

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=64,
        workers=4,
        patience=15
    )
if __name__ == "__main__":
    main()