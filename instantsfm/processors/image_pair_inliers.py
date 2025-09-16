import numpy as np
import tqdm

from instantsfm.scene.defs import ImagePair, ConfigurationType, ViewGraph
from instantsfm.utils.two_view_geometry import E_from_motion, to_matrix, sampson_error, check_cheirality, get_orientation_signum, homography_error

def score_error_homography(pair:ImagePair, images, INLIER_THRESHOLD_OPTIONS):
    image1, image2 = images[pair.image_id1], images[pair.image_id2]
    threshold = INLIER_THRESHOLD_OPTIONS['max_epipolar_error_H']
    squared_threshold = threshold**2
    score = 0.

    pts1 = image1.features[pair.matches[:, 0]]
    pts2 = image2.features[pair.matches[:, 1]]
    inliers = []
    for i in range(len(pair.matches)):
        pt1, pt2 = pts1[i], pts2[i]
        r2 = homography_error(pair.H, pt1, pt2)

        if r2 < squared_threshold:
            score += r2
            inliers.append(i)
        else:
            score += squared_threshold
    pair.inliers = np.array(inliers)
    return score

def score_error_fundamental(pair:ImagePair, images, INLIER_THRESHOLD_OPTIONS):
    epipole = np.cross(pair.F[0], pair.F[1])
    EPSILON = 1e-6
    status = np.abs(epipole) > EPSILON
    if not np.any(status):
        epipole = np.cross(pair.F[1], pair.F[2])
    
    signums = []
    image1, image2 = images[pair.image_id1], images[pair.image_id2]
    threshold = INLIER_THRESHOLD_OPTIONS['max_epipolar_error_F']
    squared_threshold = threshold**2
    score = 0.
    inliers_pre = []
    errors = []

    pts1 = image1.features[pair.matches[:, 0]]
    pts2 = image2.features[pair.matches[:, 1]]
    for i in range(len(pair.matches)):
        pt1, pt2 = pts1[i], pts2[i]
        r2 = sampson_error(pair.F, pt1, pt2)
        if r2 < squared_threshold:
            signums.append(get_orientation_signum(pair.F, epipole, pt1, pt2))
            inliers_pre.append(i)
            errors.append(r2)
        else:
            score += squared_threshold


    signums = np.array(signums)
    inliers_pre = np.array(inliers_pre)
    errors = np.array(errors)
    positive = np.sum(signums > 0)
    negative = len(signums) - positive
    is_positive = positive > negative
    if positive == negative:
        return 0.
    
    cheirality = (signums > 0) == is_positive
    pair.inliers = np.array(inliers_pre)[cheirality]
    score = np.sum(np.array(errors)[cheirality]) + np.sum(~cheirality) * squared_threshold
    return score

def score_error_essential(pair:ImagePair, cameras, images, INLIER_THRESHOLD_OPTIONS):
    qvec, tvec = pair.rotation, pair.translation
    E = E_from_motion(qvec, tvec)
    rotation_matrix = to_matrix(qvec, tvec)[:3, :3]
    epipole12 = tvec
    epipole21 = rotation_matrix @ -tvec
    if epipole12[2] < 0:
        epipole12 = -epipole12
    if epipole21[2] < 0:
        epipole21 = -epipole21
    
    threshold = INLIER_THRESHOLD_OPTIONS['max_epipolar_error_E']
    image1, image2 = images[pair.image_id1], images[pair.image_id2]
    cam1, cam2 = cameras[image1.cam_id], cameras[image2.cam_id]
    threshold = threshold * 0.5 * (1. / cam1.focal() + 1. / cam2.focal())
    squared_threshold = threshold**2
    score = 0.
    thres_epipole = np.cos(np.deg2rad(3)) + 1e-6
    thres_angle = 1 + 1e-6
    pts1 = image1.features_undist[pair.matches[:, 0]]
    pts2 = image2.features_undist[pair.matches[:, 1]]
    inliers = []
    for i in range(len(pair.matches)):
        pt1, pt2 = pts1[i], pts2[i]
        r2 = sampson_error(E, pt1, pt2)

        if r2 < squared_threshold:
            cheirality = check_cheirality(qvec, tvec, pt1, pt2, 1e-2, 100.)
            if not cheirality:
                score += squared_threshold
                continue
            diff_angle = np.dot(pt1, np.linalg.inv(rotation_matrix) @ pt2)
            if diff_angle > thres_angle:
                score += squared_threshold
                continue
            
            diff_epipole1 = np.dot(pt1, epipole21)
            diff_epipole2 = np.dot(pt2, epipole12)
            if diff_epipole1 > thres_epipole or diff_epipole2 > thres_epipole:
                score += squared_threshold
                continue
            score += r2
            inliers.append(i)
        else:
            score += squared_threshold
    pair.inliers = np.array(inliers)
    return score

def score_error(pair:ImagePair, cameras, images, INLIER_THRESHOLD_OPTIONS):
    if pair.config in [ConfigurationType.PLANAR, ConfigurationType.PANORAMIC, ConfigurationType.PLANAR_OR_PANORAMIC]:
        return score_error_homography(pair, images, INLIER_THRESHOLD_OPTIONS)
    elif pair.config == ConfigurationType.UNCALIBRATED:
        return score_error_fundamental(pair, images, INLIER_THRESHOLD_OPTIONS)
    elif pair.config == ConfigurationType.CALIBRATED:
        return score_error_essential(pair, cameras, images, INLIER_THRESHOLD_OPTIONS)
    return 0.

def ImagePairInliersCount(view_graph:ViewGraph, cameras, images, INLIER_THRESHOLD_OPTIONS):
    valid_pairs = [pair for pair in view_graph.image_pairs.values() if pair.is_valid]
    progress = tqdm.tqdm(total=len(valid_pairs))
    for i in range(len(valid_pairs)):
        score_error(valid_pairs[i], cameras, images, INLIER_THRESHOLD_OPTIONS)
        progress.update(1)
    progress.close()