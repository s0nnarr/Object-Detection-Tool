import ultralytics

from video_stream import VideoStream
from ultralytics import YOLO
import torch
import os
import cv2
from huggingface_hub import hf_hub_download
print("torch imported")
print(torch.cuda.is_available())
print("cv2 imported: ")
print(cv2.__version__)
print("ultralytics imported: ")
print(ultralytics.__version__)

#FIXME OBSOLETE CODE.

if __name__ == "__main__":
    device = "cpu" # CPU inference by default.
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_per_process_memory_fraction(0.9)# Allow PyTorch to use up to 90% of VRAM.

    model_path = hf_hub_download(
        repo_id="s0narr/aisight-model",
        filename="best.pt",
        token=os.environ.get("HF_ACCESS_TOKEN")
    )

    model = YOLO(model_path)

    # The model is version controlled and cached locally.0
    # model = YOLO("scripts/runs/detect/train10/best.pt")
    # model.to("cuda") -- LAPTOP ONLY
    # model.half() -> causes dtype conflict during layer fusion

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    stream = VideoStream(0)
    frame = stream.get_frame()
    print(frame.shape)
    while True:
        frame = stream.get_frame()
        if frame is None:
            break
        # frame = cv2.flip(frame, 1)
        results = model(frame, device=device, imgsz=640, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow("AISight", annotated_frame)
        # cv2.imwrite("test_out.jpg", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
