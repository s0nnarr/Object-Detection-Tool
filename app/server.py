import asyncio
from contextlib import asynccontextmanager
import logging
import time
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
import numpy as np
from config import config
from model import load_model
from inference import run_inference, count_by_class, decode_jpeg

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