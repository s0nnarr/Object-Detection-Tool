# import cv2
# #FIXME Currently obsolete class
# class VideoStream:
#     # Class-level variable to hold the singleton instance
#     _instance = None
#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             print("Allocating memory for object")
#             cls._instance = super(VideoStream, cls).__new__(cls) # Calls the class's constructor
#         return cls._instance
#
#     def __init__(self, source=0):
#         #TODO: Update source to also use IP Camera URLs
#         self.cap = cv2.VideoCapture(source)
#         if not self.cap.isOpened():
#             raise ValueError(f"Could not open video source: {source}")
#
#
#     def get_frame(self):
#         ret, frame = self.cap.read()
#         while True:
#             if not ret:
#                 return None
#             return frame
#
#     def release(self):
#         if self.cap.isOpened():
#             self.cap.release()
#         cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
