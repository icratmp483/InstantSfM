import numpy as np

def undistort_process(image, cam):
    features_undist = cam.img2cam(image.features)
    features_undist = np.hstack([features_undist, np.ones((features_undist.shape[0], 1))])
    image.features_undist = features_undist / np.linalg.norm(features_undist, axis=1, keepdims=True)

def UndistortImages(cameras, images):
    for image in images:
        undistort_process(image, cameras[image.cam_id])