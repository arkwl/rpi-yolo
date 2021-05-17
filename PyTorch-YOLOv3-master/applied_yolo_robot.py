import cv2
from pytorchyolo import detect, models
import time

start = time.time()
# Load the YOLO model
model = models.load_model(
  "./config/yolov3.cfg",
  "./yolov3.weights")

# Load the image as an numpy array
img = cv2.imread("/Users/arkwl/Desktop/Workstation/cs230-robot-project/train/MOT17-05/img1/000102.jpg")

# Runs the YOLO model on the image
boxes = detect.detect_image(model, img)

print(boxes)
end = time.time()
print(end - start, "seconds")
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]
