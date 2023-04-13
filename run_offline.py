import os
import sys
import numpy as np
from compare import compare
from bodyparts import get_joint_angles_from_video
sys.path.append('/home/seanzhan/projects/others/openpose/build/python')
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/home/seanzhan/projects/others/openpose/models/"
params["model_pose"] = "BODY_25"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

gt_ja = np.load('data/joint_angles.npy')

print("getting joint angles from tai chi sequence")
in_path = '/home/seanzhan/Downloads/edtaichi.mov'
out_path = 'ed_100_frames.npy' 
if not os.path.exists(out_path):
    real_ja = get_joint_angles_from_video(in_path, opWrapper, 100)
    np.save(out_path, real_ja)
else:
    real_ja = np.load(out_path)
    print("loaded joint angles")

print("comparing with ground truth data...")
acc = compare(real_ja, gt_ja)
print(f'Accuracy: {acc}%')

# acc = compare(ja, ja)

# cam_idx = 0

# vid = cv2.VideoCapture(cam_idx)
  
# while(True):
#     ret, frame = vid.read()
#     if cam_idx == 0:
#         frame = cv2.flip(frame, 1)
#     cv2.imshow('frame', frame)

#     if cv2.waitKey(33) == ord('q'):
#         print("quit")
#         break

# vid.release()
# cv2.destroyAllWindows()