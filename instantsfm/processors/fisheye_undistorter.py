import os
import cv2
import numpy as np
from tqdm import tqdm
from instantsfm.utils.read_write_model import read_cameras_text, read_images_text, read_cameras_binary, read_images_binary
import argparse
import pypose as pp

def extract_colmap_geolocation(colmap_dir, output_path):
    cameras, images = load_camera_and_image_data(colmap_dir)
    geo_locs = {}
    for image_id, image in tqdm(images.items()):
        # calculate -R^t * T
        # geo_loc = -image.qvec2rotmat().T @ image.tvec.reshape(3, 1)
        geo_loc = image.tvec
        geo_locs[image.name] = geo_loc
    with open(output_path, "w") as f:
        for image_name, geo_loc in geo_locs.items():
            f.write(f"{image_name} {geo_loc[0]} {geo_loc[1]} {geo_loc[2]}\n")

def undistort_fisheye_images(colmap_dir, image_path, output_path=None):
    cameras, images = load_camera_and_image_data(colmap_dir)

    undistorted_images = {}
    if output_path is not None:
        geo_locs_file = f"{os.path.dirname(output_path)}/geo_locs.txt"
    for image_id, image in tqdm(images.items()):
        camera = cameras[image.camera_id]
        if camera.model in ["OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"]:
            # Undistort the image using the camera parameters
            K = np.array([[camera.params[0], 0, camera.params[2]],
                          [0, camera.params[1], camera.params[3]],
                          [0, 0, 1]])
            D = camera.params[4:]

            # Placeholder for actual undistortion logic
            # This would involve reading the image, applying the undistortion, and saving the result
            undistorted_image = apply_undistortion(f"{image_path}/{image.name}", K, D, f"{output_path}/{image.name}")
            if output_path is not None:
                undistorted_images[image_id] = undistorted_image
    
    if output_path is not None:
        extract_colmap_geolocation(colmap_dir, geo_locs_file)

    return undistorted_images

def load_camera_and_image_data(colmap_dir):
    cameras_file = f"{colmap_dir}/cameras.txt"
    images_file = f"{colmap_dir}/images.txt"
    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        cameras_file = f"{colmap_dir}/cameras.bin"
        images_file = f"{colmap_dir}/images.bin"
        cameras = read_cameras_binary(cameras_file)
        images = read_images_binary(images_file)
    else:
        cameras = read_cameras_text(cameras_file)
        images = read_images_text(images_file)
    return cameras,images


def apply_undistortion(image_path, K, D, output_path=None):
    """
    Undistort a fisheye image using camera matrix and distortion coefficients.
    
    Parameters:
    image_path (str): Path to the fisheye image
    K (np.array): 3x3 camera matrix
    D (np.array): Distortion coefficients (k1, k2, k3, k4)
    
    Returns:
    np.array: Undistorted image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from the specified path")
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Generate new camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    
    # Create maps for undistortion
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, 
        D, 
        np.eye(3),  # Identity matrix for rotation
        new_camera_matrix, 
        (w, h), 
        cv2.CV_16SC2
    )
    
    # Apply the undistortion
    undistorted_img = cv2.remap(
        img, 
        map1, 
        map2, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    
    # Crop the image to remove black borders (optional)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    new_camera_matrix_cropped = new_camera_matrix.copy()
    new_camera_matrix_cropped[0, 2] -= x  # cx = cx - x
    new_camera_matrix_cropped[1, 2] -= y  # cy = cy - y

    if output_path is not None:
        cv2.imwrite(output_path, undistorted_img)
    
    return undistorted_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort fisheye images from COLMAP data.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--output_path", type=str, default="$source_path/images", help="Path to save undistorted images")
    parser.add_argument("--colmap_path", type=str, required=True, help="Path to the COLMAP directory")
    args = parser.parse_args()
    colmap_dir = args.colmap_path
    image_path = args.image_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    undistorted_images = undistort_fisheye_images(colmap_dir, image_path, output_path)
