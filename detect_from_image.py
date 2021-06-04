# import the necessary packages
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_runtime_image, detect_image
from yolov3.configs import *

from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

from skimage.measure import block_reduce

if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

tf.compat.v1.disable_eager_execution()

def resize_image(path, filename, new_width=256, new_height=256,
                              display=False):
  
  image = cv2.imread("robot_perspective_1.jpg")
  print ("original shape:", image.shape)

  pil_image = Image.open(path)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename

#yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
#yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights
with tf.Graph().as_default():
    print("[INFO] loading model...")
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
    yolo.load_weights("./checkpoints/yolov3_custom_Tiny") # use keras weights
    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    #yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
    #load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
    print("------")
    print(yolo.summary())

    print("------")

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    #vs = VideoStream(usePiCamera=True).start()
    #time.sleep(2.0)
    # start the FPS counter
    fps = FPS().start()

    image_url = "/home/pi/robot_perspective_1.jpg"
    #downloaded_image_path = download_and_resize_image(image_url, 1280, 856)
    downloaded_image_path = resize_image(image_url, "robot_perspective_1.jpg", 630, 1200)

    image = cv2.imread("robot_perspective_1.jpg")
    print("resized shape:", image.shape)
    
    # INFERENCE HAPPENS HERE
    #image = detect_runtime_image(yolo, image, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
    image = detect_image(yolo, "/home/pi/robot_perspective_1.jpg", '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
    #image = detect_image(yolo, "robot_perspective_1.jpg", '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
 
#    # OpenCV returns bounding box coordinates in (x, y, w, h) order
#    # but we need them in (top, right, bottom, left) order, so we
#    # need to do a bit of reordering
#    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # loop over the recognized faces
#    if len(boxes) > 0:
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()

