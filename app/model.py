import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from config import config
import torch 


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

def ensure_engine_file() -> None:
    if config.ENGINE_FILE_PATH.exists():
        logger.info("Found local .engine. Checking version...")
        # IF .engine file exists, but it's actually obsolete.
        
    ensure_pt_file()



def export_engine() -> None:
    logger.info(f"Exporting .engine from .pt in FP32 format for {}")