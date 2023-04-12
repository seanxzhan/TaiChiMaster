import os
import cv2
import sys
import time
import math
import numpy as np
sys.path.append('/home/seanzhan/projects/others/openpose/build/python')
from openpose import pyopenpose as op
from bodyparts import get_joint_angles_from_image

params = dict()
params["model_folder"] = "/home/seanzhan/projects/others/openpose/models/"
params["model_pose"] = "BODY_25"
# from bodyparts import get_joint_angles_from_image

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# gt_frames_path = 'data/ground_truths.npz'
# gt_frames = np.load(gt_frames_path)["arr_0"]
n_frames = 100
gt_angles = np.zeros((n_frames, 8), np.float32)

# print(gt_frames[0].shape)
# exit(0)

num_null = 0

for i in range(n_frames):
    in_path = f'data/{i}.png'
    frame = cv2.imread(in_path)
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    all_keypoints = datum.poseKeypoints
    if all_keypoints is None:
        num_null += 1
        cv2.imwrite(f'kp/{i}-notfound.png', frame)
        continue
    # gt_angles[i] = get_joint_angles_from_image(all_keypoints)
    gt_angles[i], frame = get_joint_angles_from_image(
        all_keypoints, draw=True, in_frame=frame)
    cv2.imwrite(f'kp/{i}-found.png', frame)

gt_angles = gt_angles[:n_frames - num_null]
np.save('data/joint_angles.npy', gt_angles)

