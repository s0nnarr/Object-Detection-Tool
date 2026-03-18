import ultralytics

from video_stream import VideoStream
from ultralytics import YOLO
import torch
import cv2
print("torch imported")
print(torch.cuda.is_available())
print("cv2 imported: ")
print(cv2.__version__)
print("ultralytics imported: ")
print(ultralytics.__version__)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.9) # Allow PyTorch to use up to 90% of VRAM.

    model = YOLO("best.pt")
    model.to("cuda")
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
        results = model(frame, device="cuda", imgsz=640, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow("AISight", annotated_frame)
        # cv2.imwrite("test_out.jpg", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stream.release()
