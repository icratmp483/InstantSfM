import numpy as np

from instantsfm.scene.defs import ViewGraph

def FilterRotations(view_graph:ViewGraph, images, max_angle):
    num_invalid = 0
    for pair in view_graph.image_pairs.values():
        if not pair.is_valid:
            continue
        image1 = images[pair.image_id1]
        image2 = images[pair.image_id2]
        if image1.is_registered and image2.is_registered:
            pose1 = image2.world2cam @ np.linalg.inv(image1.world2cam)
            pose2 = pair.get_cam1to2()
            inv_pose1_rot = np.linalg.inv(pose1[:3, :3])
            rotation_matrix = inv_pose1_rot @ pose2[:3, :3]
            trace = np.trace(rotation_matrix)
            cos_r = np.clip((trace - 1) / 2, -1.0, 1.0)
            angle = np.rad2deg(np.arccos(cos_r))
            if angle > max_angle:
                pair.is_valid = False
                num_invalid += 1
    print('Filtered', num_invalid, 'relative rotation with angle >', max_angle, 'degrees')

def FilterInlierNum(view_graph:ViewGraph, min_inlier_num):
    num_invalid = 0
    for pair in view_graph.image_pairs.values():
        if not pair.is_valid:
            continue
        if len(pair.inliers) < min_inlier_num:
            pair.is_valid = False
            num_invalid += 1
    print('Filtered', num_invalid, 'relative pose with inlier number <', min_inlier_num)

def FilterInlierRatio(view_graph:ViewGraph, min_inlier_ratio):
    num_invalid = 0
    for pair in view_graph.image_pairs.values():
        if not pair.is_valid:
            continue
        if len(pair.inliers) / len(pair.matches) < min_inlier_ratio:
            pair.is_valid = False
            num_invalid += 1
    print('Filtered', num_invalid, 'relative pose with inlier ratio <', min_inlier_ratio)