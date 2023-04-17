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

all_frames_paths = [
    'data/demo/1_resized.png',
    'data/demo/2_resized.png',
    'data/demo/3_resized.png',
    'data/demo/4_resized.png',
    'data/demo/5_resized.png']
# 72, 87

cam_idx = 0

vid = cv2.VideoCapture(cam_idx)
# all_acc = []

pose = 0
rf_path = all_frames_paths[pose]

while(True):
    ret, frame = vid.read()
    if cam_idx == 0:
        frame = cv2.flip(frame, 1)
    ang, drawn_frame = get_joint_angles_from_image(frame, opWrapper, draw=True)
    if ang is None:
        continue
    # cv2.imshow('frame', frame)
    gt_frame = cv2.imread(rf_path)
    # gt_frame = cv2.copyMakeBorder(gt_frame, 60, 60, 0, 0, cv2.BORDER_CONSTANT)
    gt_ang, _ = get_joint_angles_from_image(gt_frame, opWrapper)
    acc = compare.angle_similarity(ang, gt_ang)
    # all_acc.append(acc)
    # print(f'current pose accuracy: {acc}%')
    # print(drawn_frame.shape)

    frames_stacked = np.concatenate((gt_frame, drawn_frame), axis=1)

    frames_stacked = cv2.putText(
        img=frames_stacked,
        text=f'{acc*100:.2f}%',
        org=(20, 45),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(0, 0, 0),
        thickness=3
    )

    cv2.imshow('frame', frames_stacked)

    k = cv2.waitKey(33)
    if k == ord('q'):
        print("quit")
        break
    if k == ord('1'):
        print("switching to pose 1")
        rf_path = all_frames_paths[0]
    if k == ord('2'):
        print("switching to pose 2")
        rf_path = all_frames_paths[1]
    if k == ord('3'):
        print("switching to pose 3")
        rf_path = all_frames_paths[2]
    if k == ord('4'):
        print("switching to pose 4")
        rf_path = all_frames_paths[3]
    if k == ord('5'):
        print("switching to pose 5")
        rf_path = all_frames_paths[4]

vid.release()
cv2.destroyAllWindows()
