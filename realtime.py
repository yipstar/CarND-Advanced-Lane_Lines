import numpy as np
import cv2
import pickle

from threading import Thread
from imutils.video import FileVideoStream
from imutils.video import FPS

from lane_detector import LaneDetector

from perspective_transform import perspective_transform

camera_calibration = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))

lane_detector = LaneDetector()

cap = cv2.VideoCapture('project_video.mp4')

# cap = FileVideoStream("project_video.mp4").start()

fps = FPS().start()

while(cap.isOpened()):

# opened = True
# while(opened):

    ret, frame = cap.read()
    # frame = cap.read()

    result = cv2.undistort(frame, camera_calibration["mtx"], camera_calibration["dist"], None, camera_calibration["mtx"])

    result = lane_detector.process_image(result)

    cv2.imshow('frame', result)

    fps.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cap.stop()
cap.release()
cv2.destroyAllWindows()
