import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, HfApi
from ultralytics import YOLO

from app.config import config

logger = logging.getLogger("inference.model")


def remote_model_revision() -> Optional[str]:
    if not config.HF_REPO_ID:
        raise FileNotFoundError(
            "config.HF_REPO_ID missing, please update the configuration class."
        )
    try:
        info = HfApi().model_info(
            config.HF_REPO_ID,
            token=config.HF_TOKEN,
            files_metadata=True,
        )
        for s in info.siblings:
            if s.rfilename == config.HF_FILENAME and s.blob_id:
                return s.blob_id
        return info.sha
    except Exception as e:
        logger.warning("Huggingface revision check failed: %s", e)
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
    downloaded = hf_hub_download(
        repo_id=config.HF_REPO_ID,
        filename=config.HF_FILENAME,
        token=config.HF_TOKEN,
        local_dir=str(config.MODEL_DIR),
    )
    if str(config.PT_PATH) != downloaded:
        Path(downloaded).replace(config.PT_PATH)


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
            "config.HF_REPO_ID missing, please update the configuration class."
        )

    download_model()
    logger.info("Model downloaded with success.")
    write_revision(remote)
    return True


def export_engine() -> None:
    fmt = "FP16" if config.HALF else "FP32"
    model = YOLO(str(config.PT_PATH))
    logger.info("\nLoaded YOLO model.")
    logger.info("\nExporting .engine from .pt in %s format...", fmt)

    new_engine = model.export(
        format="engine",
        imgsz=config.STANDARD_IMGSZ,
        half=config.HALF,
        device=config.DEVICE,
    )
    if str(config.ENGINE_FILE_PATH) != str(new_engine):
        Path(new_engine).replace(config.ENGINE_FILE_PATH)
    logger.info("Engine ready.")


def ensure_engine_file() -> None:
    pt_updated = ensure_pt_file()
    if config.ENGINE_FILE_PATH.exists() and not pt_updated:
        logger.info("Found local .engine file.")
        return
    export_engine()


def load_model() -> YOLO:
    ensure_engine_file()
    model = YOLO(str(config.ENGINE_FILE_PATH))
    logger.info("Model loaded.")
    return model