FROM ultralytics/ultralytics:latest

WORKDIR /scripts
COPY requirements.txt .
RUN pip install roboflow python-dotenv huggingface_hub fastapi uvicorn opencv-python-headless ultralytics

COPY scripts/train.py .
LABEL authors="s0narr"

CMD ["python", "train.py"]