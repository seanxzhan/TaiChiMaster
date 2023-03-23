import numpy as np

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
    vec1 = (b - a) / np.linalg.norm(b - a)
    vec2 = (c - a) / np.linalg.norm(c - a)
    deg_rad = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    return deg_rad * 180 / np.pi

