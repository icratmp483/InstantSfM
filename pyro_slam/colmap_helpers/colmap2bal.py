import os

import numpy as np
from tqdm import tqdm
from pyro_slam.utils.ba import openGL2gtsam
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
import cv2
import pypose as pp
import torch

format_func = lambda x: f"{x:.6e}"
np.set_printoptions(formatter={'float_kind': format_func})

def colmap_to_bal(rotation_colmap, translation_colmap):
    """
    Convert COLMAP camera pose to BAL camera pose.
    
    Parameters:
    rotation_colmap (np.ndarray): 3x3 rotation matrix from COLMAP.
    translation_colmap (np.ndarray): 3x1 translation vector from COLMAP.
    
    Returns:
    rotation_bal (np.ndarray): 3x3 rotation matrix for BAL.
    translation_bal (np.ndarray): 3x1 translation vector for BAL.
    """
    T_colmap_to_bal = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=rotation_colmap.dtype, device=rotation_colmap.device)
    
    # Convert rotation matrix
    rotation_bal = T_colmap_to_bal @ rotation_colmap
    
    # Convert translation vector
    translation_bal = T_colmap_to_bal @ translation_colmap
    
    return rotation_bal, translation_bal

def colmap2bal(input_path, output_path):
    cameras = read_cameras_binary(os.path.join(input_path, "cameras.bin"))
    images = read_images_binary(os.path.join(input_path, "images.bin"))
    points3D = read_points3D_binary(os.path.join(input_path, "points3D.bin"))

    useful_points3D_ids = []
    useful_image_ids = []
    num_observations = 0

    for image_id, image in tqdm(images.items()):
        if image.camera_id not in cameras:
            continue
        useful_flag = 0
        for xy, point3D_id in zip(image.xys, image.point3D_ids):
            if point3D_id != -1:
                num_observations += 1
                useful_flag = 1
                if point3D_id not in useful_points3D_ids:
                    useful_points3D_ids.append(point3D_id)
        if useful_flag:
            useful_image_ids.append(image_id)
    num_cameras = len(useful_image_ids)
    num_points = len(useful_points3D_ids)
            

    with open(output_path, 'w') as f:
        # Write header
        f.write(f"{num_cameras} {num_points} {num_observations}\n")

        # Write observations
        for image_id in useful_image_ids:
            image = images[image_id]
            camera = cameras[image.camera_id]
            assert camera.model == "SIMPLE_RADIAL" # f, cx, cy, k
            np.testing.assert_allclose(
                np.array([camera.width / camera.params[1], camera.height / camera.params[2]]), 
                np.array([2.0, 2.0]))
            intr_matrix = np.array([[camera.params[0], 0, camera.params[1]], 
                                    [0, camera.params[0], camera.params[2]], 
                                    [0, 0, 1]], dtype=image.xys.dtype)
            dist_params = np.array([camera.params[3], 0, 0, 0], dtype=image.xys.dtype)
            # xys_ph = cv2.undistortPoints(image.xys, intr_matrix, dist_params, P=intr_matrix)
            xys = image.xys.copy()
            xys = xys - np.array([camera.params[1], camera.params[2]])[None, :]
            xys[..., 1] = xys[..., 1] * -1

            for xy, point3D_id in zip(xys, image.point3D_ids):
                if point3D_id == -1 or point3D_id not in useful_points3D_ids:
                    continue
                f.write(f"{useful_image_ids.index(image_id)} {useful_points3D_ids.index(point3D_id)}     {format_func(xy[0])} {format_func(xy[1])}\n")
        
        # Write camera parameters
        for idx, camera_id in enumerate(useful_image_ids):
            image = images[camera_id]
            camera = cameras[image.camera_id]
            assert camera.model == "SIMPLE_RADIAL" # f, cx, cy, k
            qvec = image.qvec # QW, QX, QY, QZ
            tvec = image.tvec
            pose = pp.SE3([tvec[0], tvec[1], tvec[2], qvec[1], qvec[2], qvec[3], qvec[0]])
            # Convert to BAL camera pose
            rot, t = colmap_to_bal(pose.rotation().matrix(), pose.translation())
            rot = pp.mat2SO3(rot)
            tvec = t.numpy()
            rot = rot.Log().numpy()
            # Each camera is a set of 9 parameters - R,t,f,k1 and k2. The rotation R is specified as a Rodrigues' vector.
            f.write(f"{format_func(rot[0])}\n{format_func(rot[1])}\n{format_func(rot[2])}\n{format_func(tvec[0])}\n{format_func(tvec[1])}\n{format_func(tvec[2])}\n{format_func(camera.params[0])}\n{format_func(camera.params[3])}\n{format_func(0.0)}\n")
        # Write 3D points
        for point3D_id in useful_points3D_ids:
            point3D = points3D[point3D_id]
            f.write(f"{format_func(point3D.xyz[0])}\n{format_func(point3D.xyz[1])}\n{format_func(point3D.xyz[2])}\n")

            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    colmap2bal(args.input_path, args.output_path)
"""
In COLMAP, the camera coordinate system of an image is defined in a way that the X axis points to the right, the Y axis to the bottom, and the Z axis to the front as seen from the image.

In BAL, the origin of the image is the center of the image, the positive x-axis points right, and the positive y-axis points up (in addition, in the camera coordinate system, the positive z-axis points backwards, so the camera is looking down the negative z-axis, as in OpenGL).

Write a Python function to convert a COLMAP camera pose, including rotation and translation, to the BAL camera pose. 
"""