import os
import cv2
import sys
import numpy as np
import compare
from bodyparts import get_joint_angles_from_image
sys.path.append('/home/seanzhan/projects/others/openpose/build/python')
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/home/seanzhan/projects/others/openpose/models/"
params["model_pose"] = "BODY_25"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

gt_ja = np.load('data/joint_angles.npy')

cam_idx = 0

vid = cv2.VideoCapture(cam_idx)
all_acc = []

while(True):
    ret, frame = vid.read()
    if cam_idx == 0:
        frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    ang, frame = get_joint_angles_from_image(frame, opWrapper)
    # cv2.imshow('frame', frame)
    acc = compare.get_accuracy(ang, gt_ja)
    all_acc.append(acc)
    print(f'current pose accuracy: {acc}%, overall accuracy: {np.mean(all_acc)}')

    if cv2.waitKey(33) == ord('q'):
        print("quit")
        break

vid.release()
cv2.destroyAllWindows()
