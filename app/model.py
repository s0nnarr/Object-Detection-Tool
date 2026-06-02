import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
from app.config import config
from ultralytics import YOLO
from huggingface_hub import HfApi


logger = logging.getLogger("inference.model")

def remote_model_revision() -> Optional[str]:
    if not config.HF_REPO_ID:
        raise FileNotFoundError(
            f"config.PT_PATH missing, please update the configuration class."
        )
    try:
        info = HfApi().model_info(
            config.HF_REPO_ID,
            token=config.HF_TOKEN,
            files_metadata=True
        )
        for s in info.siblings:
            if s.rfilename == config.HF_FILENAME and s.blob_id:
                return s.blob_id
        return info.sha
    except Exception as e:
        logger.warning("Huggingface revision check failed: ", e)
        return None

def local_revision() -> Optional[str]:
    if config.VERSION_MARKER.exists():
        return config.VERSION_MARKER.read_text().strip()
    return None

def write_revision(revision: Optional[str]) -> None:
    if revision:
        config.VERSION_MARKER.write_text(revision)

def download_model() -> None:

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from Huggingface.")
    try:
        download_model = hf_hub_download(repo_id=config.HF_REPO_ID, filename=config.HF_FILENAME, token=config.HF_TOKEN,
                                         local_dir=str(config.MODEL_DIR))
        if str(config.PT_PATH) != download_model:
            Path(download_model).replace(config.PT_PATH)
    except Exception as e:
        logging.warning("Something went wrong when downloading the model: ", e)



def ensure_pt_file() -> bool:
    """
    returns: true if the .pt was (re)downloaded, false otherwise.
    """
    remote = remote_model_revision() if config.CHECK_HF_VERSION else None
    local = local_revision()

    if config.PT_PATH.exists():
        if remote is None or remote == local:
            logger.info("Local .pt up to date with the remote.")
            return False
        logger.info("Newer version on Huggingface detected. Switching to model downloading.")

    if not config.HF_REPO_ID:
        raise FileNotFoundError(
            "config.PT_PATH missing, please update the configuration class."
        )

    download_model()
    logger.info("Model downloaded with success.")
    logger.info("Writing new version marker...")
    config.VERSION_MARKER.write_text(remote)
    return True

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
