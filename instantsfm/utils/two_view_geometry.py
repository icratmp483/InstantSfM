import numpy as np
from scipy.spatial.transform import Rotation as R

EPSILON = 1e-10


def E_from_motion(qvec, tvec):
    Rot = R.from_quat(qvec).as_matrix()
    E = np.array([[0, -tvec[2], tvec[1]],
                  [tvec[2], 0, -tvec[0]],
                  [-tvec[1], tvec[0], 0]])
    E = E @ Rot
    return E

def F_from_motion_and_camera(cam1, cam2, qvec, tvec):
    E = E_from_motion(qvec, tvec)
    F = cam2.get_K().T @ E @ np.linalg.inv(cam1.get_K())
    return E, F

def to_matrix(qvec, tvec):
    Rot = R.from_quat(qvec).as_matrix()
    return np.vstack([np.hstack([Rot, tvec[:, np.newaxis]]), [0, 0, 0, 1]])

def sampson_error(E, x1, x2):
    # x can be undistorted as well as distorted
    x1 = np.hstack([x1, 1]) if len(x1) == 2 else x1
    x2 = np.hstack([x2, 1]) if len(x2) == 2 else x2
    Ex1 = np.dot(E, x1) / (x1[2] + EPSILON)
    Etx2 = np.dot(E.T, x2) / (x2[2] + EPSILON)

    C = np.dot(x2, Ex1)
    Cx = Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1]
    Cy = Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1]
    return C**2 / (Cx + Cy)

def check_cheirality(qvec, tvec, x1, x2, min_depth, max_depth):
    Rot = R.from_quat(qvec).as_matrix()
    Rx1 = Rot @ x1
    a = np.dot(-Rx1, x2)
    b1 = np.dot(-Rx1, tvec)
    b2 = np.dot(tvec, x2)

    lambda1 = b1 - a * b2
    lambda2 = -a * b1 + b2
    min_depth = min_depth * (1 - a**2)
    max_depth = max_depth * (1 - a**2)
    return lambda1 > min_depth and lambda2 > min_depth and lambda1 < max_depth and lambda2 < max_depth

def get_orientation_signum(F, epipole, pt1, pt2):
    signum1 = F[0, 0] * pt2[0] + F[1, 0] * pt2[1] + F[2, 0]
    signum2 = epipole[1] - epipole[2] * pt1[1]
    return signum1 * signum2

def homography_error(H, x1, x2):
    Hx1 = H @ np.hstack([x1, 1])
    Hx1_norm = Hx1[:2] / (Hx1[2] + EPSILON)
    return np.sum((Hx1_norm - x2)**2)