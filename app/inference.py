from ultralytics import YOLO
import struct
from app.config import config
import numpy as np
from typing import Any, Optional
import cv2

HEADER_SIZE = 13 # uint32 frame_id + uint16 w + uint16 h + uint8 encoding + uint32 size

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


def decode_jpeg(payload: bytes) -> Optional[np.ndarray]:
    if len(payload) < HEADER_SIZE:
        return None

    frame_id, width, height, encoding, image_size = struct.unpack(
        ">IHHBI", payload[:HEADER_SIZE]
    )
    jpeg = payload[HEADER_SIZE:]

    n = len(jpeg)
    if n < config.MIN_PAYLOAD_SIZE or n > config.MAX_PAYLOAD_SIZE:
        return None
    if not (jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"):
        return None

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    if min(img.shape[:2]) < config.MIN_DIM:
        return None
    return img