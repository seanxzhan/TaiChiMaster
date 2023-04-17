import cv2
import sys
import numpy as np
import copy
sys.path.append('/home/seanzhan/projects/others/openpose/build/python')
from openpose import pyopenpose as op
from tqdm import tqdm
from math import floor

# params = dict()
# params["model_folder"] = "/home/seanzhan/projects/others/openpose/models/"
# params["model_pose"] = "BODY_25"
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()

nose = 0
neck = 1
rshoulder = 2
lshoulder = 5
relbow = 3
lelbow = 6
rhand = 4
lhand = 7
waist = 8
rthigh = 9
lthigh = 12
rknee = 10
lknee = 13
rfoot = 11
lfoot = 14


# linkage wise, a is between b and c. b -- a -- c
# a, b, c are 2D coordinates
def calc_joint_angle(aa, bb, cc):
    a = np.array(aa)
    b = np.array(bb)
    c = np.array(cc)
    if np.linalg.norm(b-a) == 0 or np.linalg.norm(c-a) == 0:
        return 90
    vec1 = (b - a) / np.linalg.norm(b - a)
    vec2 = (c - a) / np.linalg.norm(c - a)
    deg_rad = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    return deg_rad * 180 / np.pi


def get_joint_angles_from_video(source_path, opWrapper, cn_frames=-1):
    cap = cv2.VideoCapture(source_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cn_frames != -1:
        assert cn_frames <= n_frames
        # n_frames = cn_frames
        step_size = floor(n_frames / cn_frames)
    else:
        cn_frames = n_frames
        step_size = 1
    n_null = 0
    gt_angles = np.zeros((n_frames, 8), np.float32)
    for i in tqdm(range(min(n_frames, cn_frames))):
        frame_c = i * step_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_c)
        _, frame = cap.read()
        ang, frame = get_joint_angles_from_image(frame, opWrapper, draw=True)
        cv2.imwrite(f'ed/{i}.png', frame)
        if ang is None:
            n_null += 1
        else:
            gt_angles[i - n_null] = ang
    gt_angles = gt_angles[:cn_frames - n_null]

    print("null_frames: ", n_null)
    return gt_angles


def get_joint_angles_from_image(frame, opWrapper, draw=False):
    # opWrapper = op.WrapperPython()
    # opWrapper.configure(params)
    # opWrapper.start()

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    all_keypoints = datum.poseKeypoints
    if all_keypoints is None:
        return None, None

    # print(all_keypoints)

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

    rshoulderAng = calc_joint_angle(rshoulderC, neckC, relbowC)
    relbowAng = calc_joint_angle(relbowC, rshoulderC, rhandC)
    lshoulderAng = calc_joint_angle(lshoulderC, neckC, lelbowC)
    lelbowAng = calc_joint_angle(lelbowC, lshoulderC, lhandC)
    rthighAng = calc_joint_angle(rthighC, waistC, rkneeC)
    rkneeAng = calc_joint_angle(rkneeC, rthighC, rfootC)
    lthighAng = calc_joint_angle(lthighC, waistC, lkneeC)
    lkneeAng = calc_joint_angle(lkneeC, lthighC, lfootC)

    if draw:
        assert frame is not None
        # frame = copy.deepcopy(in_frame)
        circleColor = (256, 0, 0)
        circleRad = 1

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

        lineColor = (0, 256, 256)
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

    if not draw:
        frame = None

    return np.array([
        rshoulderAng, relbowAng, lshoulderAng, lelbowAng,
        rthighAng, rkneeAng, lthighAng, lkneeAng], dtype=np.float32),\
        frame

