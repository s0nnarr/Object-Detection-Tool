from tabnanny import verbose

from ultralytics import YOLO

from app.config import config
import numpy as np
from typing import Any, Optional


def run_inference(model: YOLO, image: np.ndarray) -> list[dict[str, Any]]:
    results = model.predict(
        source=image,
        conf=config.CONF_THRESHOLD,
        imgsz=config.STANDARD_IMGSZ,
        device=config.DEVICE,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None:
        return []
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        detections.append({
            "class_id": cls_id,
            "class_name": model.names.get(cls_id, str(cls_id)),
            "confidence": float(box.conf[0].item()),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return detections

def count_by_class(detections: list[dict[str, Any]]) -> dict[str, int]:
    # Will be deprecated.
    # Create separate class that holds the state of the class or use React's built-in state for counting each module. Speed is not the feature here.
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection["class_name"]] = counts.get(detection["class_name"], 0) + 1
    return counts


def decode_jpeg(bytes: bytes) -> Optional[np.ndarray]:
