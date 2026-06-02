FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/app
RUN apt-get update && apt-get install -y python3 python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app ./app
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]