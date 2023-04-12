import cv2
import sys
import numpy as np
from bodyparts import *
sys.path.append('/home/seanzhan/projects/others/openpose/build/python')
from openpose import pyopenpose as op

params = dict()
params["model_folder"] = "/home/seanzhan/projects/others/openpose/models/"
params["model_pose"] = "BODY_25"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

test_img_path = "test_image.jpg"
# test_img_path = "/home/seanzhan/projects/TaiChiMaster/kp/10-notfound.png"
frame = cv2.imread(test_img_path)

# gt_frames_path = 'data/ground_truths.npz'
# gt_frames = np.load(gt_frames_path)["arr_0"]
# frame = gt_frames[10]

# scale_percent = 20 # percent of original size
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100)
# dim = (width, height)
# frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

# cv2.imshow("frame", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(frame.shape)

datum = op.Datum()
datum.cvInputData = frame
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
# cv2.imshow("frame", datum.cvOutputData)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

all_keypoints = datum.poseKeypoints

_, frame = get_joint_angles_from_image(
    all_keypoints, draw=True, in_frame=frame)
cv2.imwrite('test_out.png', frame)
exit(0)

keypoints = all_keypoints[0]
noseX = keypoints[nose][0]
noseY = keypoints[nose][1]
neckX = keypoints[neck][0]
neckY = keypoints[neck][1]
lshoulderX = keypoints[lshoulder][0]
lshoulderY = keypoints[lshoulder][1]
rshoulderX = keypoints[rshoulder][0]
rshoulderY = keypoints[rshoulder][1]
lelbowX = keypoints[lelbow][0]
lelbowY = keypoints[lelbow][1]
relbowX = keypoints[relbow][0]
relbowY = keypoints[relbow][1]
lhandX = keypoints[lhand][0]
lhandY = keypoints[lhand][1]
rhandX = keypoints[rhand][0]
rhandY = keypoints[rhand][1]
waistX = keypoints[waist][0]
waistY = keypoints[waist][1]
lthighX = keypoints[lthigh][0]
lthighY = keypoints[lthigh][1]
rthighX = keypoints[rthigh][0]
rthighY = keypoints[rthigh][1]
lkneeX = keypoints[lknee][0]
lkneeY = keypoints[lknee][1]
rkneeX = keypoints[rknee][0]
rkneeY = keypoints[rknee][1]
lfootX = keypoints[lfoot][0]
lfootY = keypoints[lfoot][1]
rfootX = keypoints[rfoot][0]
rfootY = keypoints[rfoot][1]

noseC = (noseX, noseY)
neckC = (neckX, neckY)
lshoulderC = (lshoulderX, lshoulderY)
rshoulderC = (rshoulderX, rshoulderY)
lelbowC = (lelbowX, lelbowY)
relbowC = (relbowX, relbowY)
lhandC = (lhandX, lhandY)
rhandC = (rhandX, rhandY)
waistC = (waistX, waistY)
lthighC = (lthighX, lthighY)
rthighC = (rthighX, rthighY)
lkneeC = (lkneeX, lkneeY)
rkneeC = (rkneeX, rkneeY)
lfootC = (lfootX, lfootY)
rfootC = (rfootX, rfootY)

circleColor = (256, 256, 256)
circleRad = 10

cv2.circle(frame, (noseX, noseY), circleRad, circleColor)
cv2.circle(frame, (neckX, neckY), circleRad, circleColor)
cv2.circle(frame, (lshoulderX, lshoulderY), circleRad, circleColor)
cv2.circle(frame, (rshoulderX, rshoulderY), circleRad, circleColor)
cv2.circle(frame, (lelbowX, lelbowY), circleRad, circleColor)
cv2.circle(frame, (relbowX, relbowY), circleRad, circleColor)
cv2.circle(frame, (lhandX, lhandY), circleRad, circleColor)
cv2.circle(frame, (rhandX, rhandY), circleRad, circleColor)
cv2.circle(frame, (waistX, waistY), circleRad, circleColor)
cv2.circle(frame, (lthighX, lthighY), circleRad, circleColor)
cv2.circle(frame, (rthighX, rthighY), circleRad, circleColor)
cv2.circle(frame, (lkneeX, lkneeY), circleRad, circleColor)
cv2.circle(frame, (rkneeX, rkneeY), circleRad, circleColor)
cv2.circle(frame, (lfootX, lfootY), circleRad, circleColor)
cv2.circle(frame, (rfootX, rfootY), circleRad, circleColor)

lineColor = (256, 256, 256)
lineWidth = 3

cv2.line(frame, noseC, neckC, lineColor, lineWidth)
cv2.line(frame, neckC, lshoulderC, lineColor, lineWidth)
cv2.line(frame, neckC, rshoulderC, lineColor, lineWidth)
cv2.line(frame, lshoulderC, lelbowC, lineColor, lineWidth)
cv2.line(frame, lelbowC, lhandC, lineColor, lineWidth)
cv2.line(frame, rshoulderC, relbowC, lineColor, lineWidth)
cv2.line(frame, relbowC, rhandC, lineColor, lineWidth)
cv2.line(frame, neckC, waistC, lineColor, lineWidth)
cv2.line(frame, waistC, lthighC, lineColor, lineWidth)
cv2.line(frame, waistC, rthighC, lineColor, lineWidth)
cv2.line(frame, lthighC, lkneeC, lineColor, lineWidth)
cv2.line(frame, lkneeC, lfootC, lineColor, lineWidth)
cv2.line(frame, rthighC, rkneeC, lineColor, lineWidth)
cv2.line(frame, rkneeC, rfootC, lineColor, lineWidth)

rshoulderAng = calc_joint_angle(rshoulderC, neckC, relbowC)
relbowAng = calc_joint_angle(relbowC, rshoulderC, rhandC)
lshoulderAng = calc_joint_angle(lshoulderC, neckC, lelbowC)
lelbowAng = calc_joint_angle(lelbowC, lshoulderC, lhandC)
rthighAng = calc_joint_angle(rthighC, waistC, rkneeC)
rkneeAng = calc_joint_angle(rkneeC, rthighC, rfootC)
lthighAng = calc_joint_angle(lthighC, waistC, lkneeC)
lkneeAng = calc_joint_angle(lkneeC, lthighC, lfootC)

print(calc_joint_angle(neckC, noseC, rshoulderC))

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
