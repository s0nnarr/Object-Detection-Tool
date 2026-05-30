import os 
from pathlib import Path

class Config:
    MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/models"))
    PT_PATH = MODEL_DIR / os.getenv("PT_NAME", "best.pt")
    ENGINE_FILE_PATH = MODEL_DIR / os.getenv("ENGINE_NAME", "best.engine")

    HF_REPO_ID = os.getenv("HF_REPO_ID")
    HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
    HF_TOKEN = os.getenv("HF_TOKEN") #HuggingFace client secret. 

    STANDARD_IMGSZ = int(os.getenv("STANDARD_IMGSZ", "640"))
    HALF = os.getenv("HALF", "true").lower() == "true"
    DEVICE = int(os.getenv("DEVICE", "0")) # Device number, by default 0 (for laptop webcam)
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.7"))
 
    MAX_PAYLOAD_SIZE = int(os.getenv("MAX_PAYLOAD_SIZE", "2000000"))
    MIN_PAYLOAD_SIZE = int(os.getenv("MIN_PAYLOAD_SIZE", "5000"))
    MIN_DIM = int(os.getenv("MIN_DIM", "160"))
 
    MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "1"))
    CHECK_HF_VERSION = os.getenv("CHECK_HF_VERSION", "true").lower() == "true"
    VERSION_MARKER = MODEL_DIR / ".model_revision"


config = Config()
