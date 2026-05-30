import logging
from pathlib import Path

from fontTools.varLib.mutator import half
from huggingface_hub import hf_hub_download
from tensorflow.python.framework import device

from config import config
from ultralytics import YOLO

logger = logging.getLogger("inference.model")

def ensure_pt_file() -> None:
    if config.PT_PATH.exists():
        logger.info("Found local .pt: %s", config.PT_PATH)
        return
    
    if not config.HF_REPO_ID:
        raise FileNotFoundError(
            f"config.PT_PATH missing, please update the configuration class."
        )
    
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from Huggingface")
    download_model = hf_hub_download(repo_id=config.HF_REPO_ID, filename=config.HF_FILENAME, token=config.HF_TOKEN, local_dir=str(config.MODEL_DIR))
    
    if str(config.PT_PATH) != download_model:
        Path(download_model).replace(config.PT_PATH)


def export_engine() -> None:
    logger.info(f"Exporting .engine from .pt in FP32 format for RTX 5090...")
    model = YOLO(str(config.PT_PATH))
    new_engine = model.export(format="engine", imgsz=config.STANDARD_IMGSZ, half=config.HALF, device=config.DEVICE)
    if str(config.ENGINE_FILE_PATH) != str(new_engine):
        Path(new_engine).replace(config.ENGINE_FILE_PATH)
    logger.info("Engine ready.")

def ensure_engine_file() -> None:
    pt_updated = ensure_pt_file()
    if config.ENGINE_FILE_PATH.exists() and not pt_updated:
        logger.info("Found local .pt file.")
        return
    export_engine()

def load_model() -> YOLO:
    ensure_engine_file()
    model = YOLO(str(config.ENGINE_FILE_PATH))
    logger.info("Model loaded.")
    return model
