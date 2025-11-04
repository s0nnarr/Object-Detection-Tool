from video_stream import VideoStream
import cv2

if __name__ == "__main__":
    stream = VideoStream(0)
    while True:
        frame = stream.get_frame()
        if frame is None:
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow("AISight", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stream.release()
