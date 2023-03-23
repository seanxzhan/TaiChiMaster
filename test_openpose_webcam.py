import os
import cv2
import sys
import time
import math
import numpy as np
sys.path.append('/home/seanzhan/projects/others/openpose/build/python')
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/home/seanzhan/projects/others/openpose/models/"
params["model_pose"] = "BODY_25"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cam_idx = 0
vid = cv2.VideoCapture(cam_idx)
  
while(True):
    ret, frame = vid.read()
    if cam_idx == 0:
        frame = cv2.flip(frame, 1)
    # cv2.imshow('frame', frame)

    # Process Image
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("frame", datum.cvOutputData)

    if cv2.waitKey(33) == ord('q'):
        print("quit")
        break

vid.release()
cv2.destroyAllWindows()