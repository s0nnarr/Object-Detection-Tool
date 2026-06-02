import logging
from idlelib.pathbrowser import PathBrowser

from ultralytics import YOLO
import os
from roboflow import Roboflow
from huggingface_hub import hf_hub_download, HfApi, upload_file
from pathlib import Path

# Download the Huggingface model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

def download_dataset() -> str:
    # model = YOLO("sku_baseline.pt")
    logger.info("Beginning dataset download")
    try:
        rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
        rf_workspace = rf.workspace("stancus-workspace").project("my-first-project-uj5as")
        dataset = rf_workspace.version(int(os.environ.get("RF_VERSION", "1"))).download("yolov8") # Downloads the dataset in yolov8 format. Anything above (10, 11, 26) uses the same format as v8.
    except Exception as e:
        logger.error(e)

    return f"{dataset.location}/data.yaml"


def download_base_model() -> str:
    model_path = hf_hub_download(
        repo_id=os.environ["HF_REPO_ID"],
        filename=os.environ["HF_FILENAME"],
        token=os.environ.get("HF_ACCESS_TOKEN")
    )
    logger.info("Downloaded base model.")
    return model_path

def upload_result(weights_path: str) -> None:
    repo_id = os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo_id:
        logger.info("HF_UPLOAD_REPO_ID not set, skipping upload.")
        return

    logger.info(f"Uploading {weights_path.name} to {repo_id}", weights_path.name, repo_id)
    upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo=weights_path.name,
        repo_id=repo_id,
        token=os.environ["HF_ACCESS_TOKEN"],
        commit_message="Training run upload",
    )
    logger.info("Upload done.")


def main():
    logger.info("Initializing training...")

    model_path = download_base_model()
    model = YOLO(model_path)
    data_yaml = download_dataset()

    results = model.train(
        data=data_yaml,
        epochs=int(os.environ.get("EPOCHS", "100")),
        imgsz=int(os.environ.get("IMGSZ", "640")),
        batch=int(os.environ.get("BATCH", "64")),
        workers=int(os.environ.get("WORKERS", "4")),
        patience=int(os.environ.get("PATIENCE", "15")),

        project="/workspace/runs",
        name="train",
        exist_ok=True,

        # augmentations
        fliplr=float(os.environ.get("AUG_FLIPLR", "0.5")),
        mosaic=float(os.environ.get("AUG_MOSAIC", "1.0")),
        hsv_h=float(os.environ.get("AUG_HSV_H", "0.015")),
        hsv_s=float(os.environ.get("AUG_HSV_S", "0.7")),
        hsv_v=float(os.environ.get("AUG_HSV_V", "0.4")),
        degrees=float(os.environ.get("AUG_DEGREES", "10.0")),
        translate=float(os.environ.get("AUG_TRANSLATE", "0.1")),
        scale=float(os.environ.get("AUG_SCALE", "0.5")),
    )

    best_weights = Path("/workspace/runs/train/weights/best.pt")
    if best_weights.exists():
        upload_result(best_weights)
    else:
        logger.error(f"best.pt not found at {best_weights}")


if __name__ == "__main__":
    main()