import asyncio
from contextlib import asynccontextmanager
import logging
import time
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
import numpy as np
from app.config import config
from app.model import load_model
from app.inference import decode_jpeg, run_inference, count_by_class
state: dict = {}
logger = logging.getLogger("server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    state["sem"] = asyncio.Semaphore(config.MAX_CONCURRENT_INFERENCE)

    dummy = np.zeros((config.STANDARD_IMGSZ, config.STANDARD_IMGSZ, 3), dtype=np.uint8)
    await asyncio.to_thread(run_inference, state["model"], dummy)
    logger.info("Warmup done. Server ready.")
    yield()
    state.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/inference")
async def inference_ws(websocket: WebSocket, session_id: str | None = Query(default=None)):
    await websocket.accept()
    model = state["model"]
    sem: asyncio.Semaphore = state["sem"]
    frame_id = 0

    await websocket.send_json({
        "type": "connection_ready",
        "session_id": session_id,
    })
    try:
        while True:
            payload_bytes = await websocket.receive_bytes()
            frame_id += 1
            # t0 = time.time()

            image = decode_jpeg(payload_bytes)
            if image is None:
                await websocket.send_json({
                    "type": "error_payload",
                    "frame_id": frame_id,
                    "message": "Payload is either invalid or corrupt"
                })
                continue
            async with sem:
                detections = await asyncio.to_thread(run_inference, model, image)
            await websocket.send_text(json.dumps({
                "type": "detections",
                "session_id": session_id,
                "frame_id": frame_id,
                "detections": detections,
                "counts": count_by_class(detections)
            }))

    except WebSocketDisconnect:
        logger.info(f"Disconnected Websockets. Session ID: {session_id}")
    except Exception as e:
        logger.exception(f"Websockets error in session ID: {session_id}:", e)

    except Exception as e:
        logger.exception("Error occured during inference: ", e)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass