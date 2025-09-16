import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

from instantsfm.utils.read_write_model import write_next_bytes

def bilinear_interpolate(image, x, y):
    h, w, c = image.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return [-1, -1, -1]

    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

    R1 = (x2 - x) * image[y1, x1] + (x - x1) * image[y1, x2]
    R2 = (x2 - x) * image[y2, x1] + (x - x1) * image[y2, x2]
    P = (y2 - y) * R1 + (y - y1) * R2

    return P

class point3d:
    def __init__(self, **kwargs):
        self.xyz = np.zeros(3)
        self.color = np.zeros(3)
        self.error = -1.
        self.track_elements = [] # track_element = (image_id, point2d_idx)
        for key, val in kwargs.items():
            setattr(self, key, val)

class Reconstruction:
    def __init__(self, **kwargs):
        self.cameras = {}
        self.images = {}
        self.point3d = {}
        for key, val in kwargs.items():
            setattr(self, key, val)

    def ExtractColorsForAllImages(self, image_path):
        color_sums = {}
        color_counts = {}

        for image_id, image in self.images.items():
            if not os.path.exists(os.path.join(image_path, image.filename)):
                continue
            bitmap = cv2.imread(os.path.join(image_path, image.filename))
            bitmap = cv2.cvtColor(bitmap, cv2.COLOR_BGR2RGB)
            
            for idx, point2d in enumerate(image.features):
                if np.all(image.point3d_ids[idx] == -1):
                    continue
                x, y = point2d - 0.5
                color = bilinear_interpolate(bitmap, x, y)
                if color[0] == -1:
                    continue
                if image.point3d_ids[idx] in color_sums:
                    color_sums[image.point3d_ids[idx]] += color
                    color_counts[image.point3d_ids[idx]] += 1
                else:
                    color_sums[image.point3d_ids[idx]] = color
                    color_counts[image.point3d_ids[idx]] = 1
        
        for track_id, point3d in self.point3d.items():
            if track_id in color_sums:
                color = color_sums[track_id] / color_counts[track_id]
                point3d.color = [int(c) for c in color]
            else:
                point3d.color = [0, 0, 0]

    def WriteCamerasBinary(self, filepath):
        """
        reimplemented from pyglomap/read_write_model.py
        same as two functions below
        """
        with open(filepath, "wb") as fid:
            if isinstance(self.cameras, dict):
                cameras_list = list(self.cameras.values())
            else:
                cameras_list = self.cameras
            write_next_bytes(fid, len(cameras_list), "Q")
            for cam in cameras_list:
                model_id = cam.model_id.value
                camera_properties = [cam.id, model_id, cam.width, cam.height]
                write_next_bytes(fid, camera_properties, "iiQQ")
                for p in cam.params:
                    write_next_bytes(fid, float(p), "d")

    def WriteImagesBinary(self, filepath):
        with open(filepath, "wb") as fid:
            if isinstance(self.images, dict):
                images_list = list(self.images.values())
            else:
                images_list = self.images
            write_next_bytes(fid, len(images_list), "Q")
            for img in images_list:
                write_next_bytes(fid, img.id, "i")
                tvec = img.world2cam[:3, 3]
                qvec = R.from_matrix(img.world2cam[:3, :3]).as_quat()
                write_next_bytes(fid, [qvec[3], *qvec[:3]], "dddd")
                write_next_bytes(fid, tvec.tolist(), "ddd")
                write_next_bytes(fid, img.cam_id, "i")
                for char in img.filename:
                    write_next_bytes(fid, char.encode("utf-8"), "c")
                write_next_bytes(fid, b"\x00", "c")
                point3D_ids = img.point3d_ids[img.point3d_ids != -1]
                write_next_bytes(fid, len(point3D_ids), "Q")
                xys = img.features[img.point3d_ids != -1]
                for xy, p3d_id in zip(xys, point3D_ids):
                    write_next_bytes(fid, [*xy, p3d_id], "ddq")

    def WritePoints3DBinary(self, filepath):
        with open(filepath, "wb") as fid:
            write_next_bytes(fid, len(self.point3d), "Q")
            for pt_id, pt in self.point3d.items():
                write_next_bytes(fid, pt_id, "Q")
                write_next_bytes(fid, pt.xyz.tolist(), "ddd")
                write_next_bytes(fid, pt.color, "BBB")
                write_next_bytes(fid, pt.error, "d")
                track_length = len(pt.track_elements)
                write_next_bytes(fid, track_length, "Q")
                for image_id, point2D_id in pt.track_elements:
                    write_next_bytes(fid, [image_id, point2D_id], "ii")

    def WriteBinary(self, path):
        self.WriteCamerasBinary(os.path.join(path, 'cameras.bin'))
        self.WriteImagesBinary(os.path.join(path, 'images.bin'))
        self.WritePoints3DBinary(os.path.join(path, 'points3D.bin'))