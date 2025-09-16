from queue import Queue
import numpy as np
from enum import Enum
import cv2
from typing import List, Optional, Union, Dict, Set
from scipy.spatial.transform import Rotation as R

class Image:
    def __init__(
        self,
        id: int = -1,
        cam_id: int = -1,
        filename: str = "",
        is_registered: bool = False,
        cluster_id: int = -1,
        world2cam: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        features_undist: Optional[np.ndarray] = None,
        point3d_ids: Optional[List[int]] = None,
        num_points3d: int = 0
    ):
        self.id = id
        self.cam_id = cam_id
        self.filename = filename
        self.is_registered = is_registered
        self.cluster_id = cluster_id
        self.world2cam = world2cam if world2cam is not None else np.eye(4)
        self.features = features if features is not None else []
        self.features_undist = features_undist if features_undist is not None else []
        self.point3d_ids = point3d_ids if point3d_ids is not None else []
        self.num_points3d = num_points3d
    
    def center(self):
        return self.world2cam[:3, :3].T @ -self.world2cam[:3, 3]
    
    def axis_angle(self):
        return R.from_matrix(self.world2cam[:3, :3]).as_rotvec()
    
class ConfigurationType(Enum):
    UNDEFINED = 0
    DEGENERATE = 1
    CALIBRATED = 2
    UNCALIBRATED = 3
    PLANAR = 4
    PANORAMIC = 5
    PLANAR_OR_PANORAMIC = 6
    WATERMARK = 7
    MULTIPLE = 8
    
class ImagePair:
    def __init__(
        self,
        image_id1: int = -1,
        image_id2: int = -1,
        is_valid: bool = True,
        weight: float = 0.0,
        E: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        inliers: Optional[List] = None,
        config: ConfigurationType = ConfigurationType.UNDEFINED
    ):
        self.image_id1 = image_id1
        self.image_id2 = image_id2
        self.is_valid = is_valid
        self.weight = weight
        self.E = E if E is not None else np.eye(3)
        self.F = F if F is not None else np.eye(3)
        self.H = H if H is not None else np.eye(3)
        # below are in the form of image 1 to image 2
        self.rotation = rotation if rotation is not None else np.array([1, 0, 0, 0])
        self.translation = translation if translation is not None else np.zeros(3)
        self.inliers = inliers if inliers is not None else []
        self.config = config

    def set_cam1to2(self, cam1to2: np.ndarray) -> None:
        rotation_matrix = cam1to2[:3, :3]
        self.rotation = R.from_matrix(rotation_matrix).as_quat(canonical=False)
        self.translation = cam1to2[:3, 3]
    
    def get_cam1to2(self) -> np.ndarray:
        rotation_matrix = R.from_quat(self.rotation).as_matrix()
        return np.vstack([np.hstack([rotation_matrix, self.translation[:, np.newaxis]]), [0, 0, 0, 1]])

C_MAX_INT = 2**31 - 1
def PairId2Ids(pair_id):
    return (pair_id%C_MAX_INT, pair_id//C_MAX_INT)

def PairId2IdsInversed(pair_id):
    return (pair_id//C_MAX_INT, pair_id%C_MAX_INT)

def Ids2PairId(id1, id2):
    return (id1*C_MAX_INT + id2 if id1 < id2 else id2*C_MAX_INT + id1)

# Camera models, note that in some models we ignore the single fourth k parameter, following opencv's convention.
# While reading from database and initializing cameras, we still keep all parameters for consistency, though some are not used.
class CameraModelId(Enum):
    INVALID = -1
    SIMPLE_PINHOLE = 0
    PINHOLE = 1
    SIMPLE_RADIAL = 2
    RADIAL = 3
    OPENCV = 4
    OPENCV_FISHEYE = 5
    FULL_OPENCV = 6
    FOV = 7
    SIMPLE_RADIAL_FISHEYE = 8
    RADIAL_FISHEYE = 9
    THIN_PRISM_FISHEYE = 10

def get_camera_model_info(model_id):
    # focal refers to which parameters are focal length, optimize refers to which parameters should be optimized (all except principal point)
    if model_id == CameraModelId.SIMPLE_PINHOLE:
        return {'name': 'SIMPLE_PINHOLE', 'num_params': 3, 'focal': [0], 'pp': [1, 2], 'k': [], 'p': [], 'omega': [], 'sx': [], 'optimize': [0]}
    elif model_id == CameraModelId.PINHOLE:
        return {'name': 'PINHOLE', 'num_params': 4, 'focal': [0, 1], 'pp': [2, 3], 'k': [], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 1]}
    elif model_id == CameraModelId.SIMPLE_RADIAL:
        return {'name': 'SIMPLE_RADIAL', 'num_params': 4, 'focal': [0], 'pp': [1, 2], 'k': [3], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3]}
    elif model_id == CameraModelId.RADIAL:
        return {'name': 'RADIAL', 'num_params': 5, 'focal': [0], 'pp': [1, 2], 'k': [3, 4], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3, 4]}
    elif model_id == CameraModelId.OPENCV:
        return {'name': 'OPENCV', 'num_params': 8, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5], 'p': [6, 7], 'omega': [], 'sx': [], 'optimize': [0, 1, 4, 5, 6, 7]}
    elif model_id == CameraModelId.OPENCV_FISHEYE:
        return {'name': 'OPENCV_FISHEYE', 'num_params': 8, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5, 6, 7], 'omega': [], 'sx': [], 'optimize': [0, 1, 4, 5, 6, 7]}
    elif model_id == CameraModelId.FULL_OPENCV:
        return {'name': 'FULL_OPENCV', 'num_params': 12, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5, 8, 9, 10, 11], 'p': [6, 7], 'omega': [], 'sx': [], 'optimize': [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]}
    elif model_id == CameraModelId.FOV:
        return {'name': 'FOV', 'num_params': 5, 'focal': [0, 1], 'pp': [2, 3], 'k': [], 'p': [], 'omega': [4], 'sx': [], 'optimize': [0, 1, 4]}
    elif model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
        return {'name': 'SIMPLE_RADIAL_FISHEYE', 'num_params': 4, 'focal': [0], 'pp': [1, 2], 'k': [3], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3]}
    elif model_id == CameraModelId.RADIAL_FISHEYE:
        return {'name': 'RADIAL_FISHEYE', 'num_params': 5, 'focal': [0], 'pp': [1, 2], 'k': [3, 4], 'p': [], 'omega': [], 'sx': [], 'optimize': [0, 3, 4]}
    elif model_id == CameraModelId.THIN_PRISM_FISHEYE:
        return {'name': 'THIN_PRISM_FISHEYE', 'num_params': 12, 'focal': [0, 1], 'pp': [2, 3], 'k': [4, 5, 8, 9], 'p': [6, 7], 'omega': [], 'sx': [10, 11], 'optimize': [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]}
    else:
        raise NotImplementedError

class Camera:
    def __init__(
        self,
        id: int = -1,
        model_id: CameraModelId = CameraModelId.INVALID,
        width: int = 0,
        height: int = 0,
        params: Optional[List[float]] = None,
        has_prior_focal_length: bool = False,
        principal_point: Optional[np.ndarray] = None,
        focal_length: Optional[np.ndarray] = None
    ):
        self.id = id
        self.model_id = model_id
        self.width = width
        self.height = height
        self.has_prior_focal_length = has_prior_focal_length
        self.principal_point = principal_point if principal_point is not None else np.zeros(2)
        self.focal_length = focal_length if focal_length is not None else np.zeros(2)
        
        # Initialize distortion parameters
        self.k: List[float] = [0.0]
        self.p: np.ndarray = np.zeros(2)
        self.omega: float = 0.0
        self.sx: np.ndarray = np.zeros(2)
        
        # Set params and initialize camera parameters
        if params is not None:
            self.set_params(params)
        else:
            self.params: List[float] = []

    def focal(self) -> float:
        return float(np.mean(self.focal_length))

    def set_params(self, params: List[float]) -> None:
        self.params = params
        if self.model_id == CameraModelId.SIMPLE_PINHOLE:
            assert len(params) == 3
            self.focal_length = np.array([params[0], params[0]])
            self.principal_point = np.array([params[1], params[2]])
        elif self.model_id == CameraModelId.PINHOLE:
            assert len(params) == 4
            self.focal_length = np.array([params[0], params[1]])
            self.principal_point = np.array([params[2], params[3]])
        elif self.model_id == CameraModelId.SIMPLE_RADIAL:
            assert len(params) == 4
            self.focal_length = np.array([params[0], params[0]])
            self.principal_point = np.array([params[1], params[2]])
            self.k = [params[3]]
        elif self.model_id == CameraModelId.RADIAL:
            assert len(params) == 5
            self.focal_length = np.array([params[0], params[0]])
            self.principal_point = np.array([params[1], params[2]])
            self.k = [params[3], params[4]]
        elif self.model_id == CameraModelId.OPENCV:
            assert len(params) == 8
            self.focal_length = np.array([params[0], params[1]])
            self.principal_point = np.array([params[2], params[3]])
            self.k = [params[4], params[5]]
            self.p = np.array([params[6], params[7]])
        elif self.model_id == CameraModelId.OPENCV_FISHEYE:
            assert len(params) == 8
            self.focal_length = np.array([params[0], params[1]])
            self.principal_point = np.array([params[2], params[3]])
            self.k = [params[4], params[5], params[6], params[7]]
        elif self.model_id == CameraModelId.FULL_OPENCV:
            assert len(params) == 12
            self.focal_length = np.array([params[0], params[1]])
            self.principal_point = np.array([params[2], params[3]])
            self.k = [params[4], params[5], params[8], params[9], params[10], params[11]]
            self.p = np.array([params[6], params[7]])
        elif self.model_id == CameraModelId.FOV:
            assert len(params) == 5
            self.focal_length = np.array([params[0], params[1]])
            self.principal_point = np.array([params[2], params[3]])
            self.omega = params[4]
        elif self.model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
            assert len(params) == 4
            self.focal_length = np.array([params[0], params[0]])
            self.principal_point = np.array([params[1], params[2]])
            self.k = [params[3]]
        elif self.model_id == CameraModelId.RADIAL_FISHEYE:
            assert len(params) == 5
            self.focal_length = np.array([params[0], params[0]])
            self.principal_point = np.array([params[1], params[2]])
            self.k = [params[3], params[4]]
        elif self.model_id == CameraModelId.THIN_PRISM_FISHEYE:
            assert len(params) == 12
            self.focal_length = np.array([params[0], params[1]])
            self.principal_point = np.array([params[2], params[3]])
            self.k = [params[4], params[5], params[8], params[9]]
            self.p = np.array([params[6], params[7]])
            self.sx = np.array([params[10], params[11]])
        else:
            raise NotImplementedError
    
    def get_K(self):
        return np.array([[self.focal_length[0], 0, self.principal_point[0]],
                         [0, self.focal_length[1], self.principal_point[1]],
                         [0, 0, 1]])
    
    def fisheye_from_normal(self, uv):
        r = np.linalg.norm(uv, axis=-1, keepdims=True)
        r = np.clip(r, 1e-8, None) # avoid division by zero
        theta = np.arctan(r)
        return uv * theta / r
    
    def normal_from_fisheye(self, uv):
        theta = np.linalg.norm(uv, axis=-1, keepdims=True)
        theta_cos_theta = theta * np.cos(theta)
        return uv * np.sin(theta) / theta_cos_theta
    
    def Distortion(self, uv):
        # this distort function can treat batchify uv as well as single uv input
        if self.model_id == CameraModelId.SIMPLE_RADIAL:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            return uv * self.k[0] * r2
        elif self.model_id == CameraModelId.RADIAL:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            return uv * self.k[0] * r2 + uv * self.k[1] * r2**2
        elif self.model_id == CameraModelId.OPENCV:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            uv_ = np.expand_dims(uv[..., 0] * uv[..., 1], axis=-1)
            radial = self.k[0] * r2 + self.k[1] * r2**2
            d = uv * radial + 2 * self.p * uv_
            d += self.p[::-1] * (r2 + 2 * uv**2)
            return d
        elif self.model_id == CameraModelId.OPENCV_FISHEYE:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            radial = self.k[0] * r2 + self.k[1] * r2**2 + self.k[2] * r2**3 # ignore k3
            return uv * radial
        elif self.model_id == CameraModelId.FULL_OPENCV:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            uv_ = np.expand_dims(uv[..., 0] * uv[..., 1], axis=-1)
            radial = (1 + self.k[0] * r2 + self.k[1] * r2**2 + self.k[2] * r2**3) / (1 + self.k[3] * r2 + self.k[4] * r2**2 + self.k[5] * r2**3) - 1
            d = uv * radial + 2 * self.p * uv_
            d += self.p[::-1] * (r2 + 2 * uv**2)
            return d
        elif self.model_id == CameraModelId.FOV:
            omega = self.omega
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            omega2 = omega**2
            epsilon = 1e-4
            if omega2 < epsilon:
                factor = (omega2 * r2) / 3 - omega2 / 12 + 1
            else:
                factor = np.zeros_like(r2)
                r2_mask = r2 < epsilon
                tan_half_omega = np.tan(omega / 2)
                factor[r2_mask] = (-2 * tan_half_omega * (4 * r2[r2_mask] * tan_half_omega**2 - 3)) / (3 * omega)
                r2_mask_inv = ~r2_mask
                radius = np.sqrt(r2[r2_mask_inv])
                numerator = np.arctan(radius * 2 * np.tan(omega / 2))
                factor[r2_mask_inv] = numerator / (radius * omega)
            return uv * factor
        elif self.model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            return uv * self.k[0] * r2
        elif self.model_id == CameraModelId.RADIAL_FISHEYE:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            return uv * self.k[0] * r2 + uv * self.k[1] * r2**2
        elif self.model_id == CameraModelId.THIN_PRISM_FISHEYE:
            r2 = np.sum(uv**2, axis=-1, keepdims=True)
            uv_ = np.expand_dims(uv[..., 0] * uv[..., 1], axis=-1)
            radial = self.k[0] * r2 + self.k[1] * r2**2 + self.k[2] * r2**3 # ignore k3
            d = uv * radial + 2 * self.p * uv_
            d += self.p[::-1] * (r2 + 2 * uv**2)
            d += self.sx * r2
            return d
        else:
            raise NotImplementedError

    def img2cam(self, xy):
        # currently only support a few models
        # cv2.undistortPoints coeffs: k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]
        # some models are not supported by cv2.undistortPoints, so we need to implement the undistortion by ourselves
        if self.model_id == CameraModelId.SIMPLE_PINHOLE:
            return (xy - self.principal_point) / self.focal()
        elif self.model_id == CameraModelId.PINHOLE:
            return (xy - self.principal_point) / self.focal_length
        elif self.model_id == CameraModelId.SIMPLE_RADIAL:
            dist_coeffs = np.array([self.k[0], 0, 0, 0])
            return cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
        elif self.model_id == CameraModelId.RADIAL:
            dist_coeffs = np.array([self.k[0], self.k[1], 0, 0])
            return cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
        elif self.model_id == CameraModelId.OPENCV:
            dist_coeffs = np.array([self.k[0], self.k[1], self.p[0], self.p[1]])
            return cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
        elif self.model_id == CameraModelId.OPENCV_FISHEYE:
            dist_coeffs = np.array([self.k[0], self.k[1], 0, 0, self.k[2]]) # ignore k3
            uv = cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
            return self.normal_from_fisheye(uv)
        elif self.model_id == CameraModelId.FULL_OPENCV:
            dist_coeffs = np.array([self.k[0], self.k[1], self.p[0], self.p[1], self.k[2], self.k[3], self.k[4], self.k[5]])
            return cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
        elif self.model_id == CameraModelId.FOV:
            omega = self.omega
            r2 = np.expand_dims(np.sum(xy**2, axis=-1), axis=-1)
            omega2 = omega**2
            epsilon = 1e-4
            if omega2 < epsilon:
                factor = (omega2 * r2) / 3 - omega2 / 12 + 1
            else:
                r2_mask = r2 < epsilon
                factor = np.zeros_like(r2)
                factor[r2_mask] = (omega * (omega2 * r2[r2_mask] + 3)) / (6 * np.tan(omega / 2))
                r2_mask_inv = ~r2_mask
                radius = np.sqrt(r2[r2_mask_inv])
                numerator = np.tan(radius * omega)
                factor[r2_mask_inv] = numerator / (radius * 2 * np.tan(omega / 2))
            uv = (xy - self.principal_point) / self.focal_length
            return uv * factor
        elif self.model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
            dist_coeffs = np.array([self.k[0], 0, 0, 0])
            uv = cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
            return self.normal_from_fisheye(uv)
        elif self.model_id == CameraModelId.RADIAL_FISHEYE:
            dist_coeffs = np.array([self.k[0], self.k[1], 0, 0])
            uv = cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
            return self.normal_from_fisheye(uv)
        elif self.model_id == CameraModelId.THIN_PRISM_FISHEYE:
            dist_coeffs = np.array([self.k[0], self.k[1], self.p[0], self.p[1], self.k[2], 0, 0, 0, self.sx[0], self.sx[1], 0, 0]) # ignore k3
            uv = cv2.undistortPoints(np.expand_dims(xy, axis=1), self.get_K(), dist_coeffs).reshape(-1, 2)
            return self.normal_from_fisheye(uv)
        else:
            raise NotImplementedError
        
    def cam2img(self, uvw):
        pp = self.principal_point
        f = np.mean(self.focal_length)
        ff = self.focal_length
        uv = uvw[..., :2] / (np.expand_dims(uvw[..., 2], axis=-1) + 1e-10)
        if self.model_id == CameraModelId.SIMPLE_PINHOLE:
            return uv * f + pp
        elif self.model_id == CameraModelId.PINHOLE:
            return uv * ff + pp
        elif self.model_id == CameraModelId.SIMPLE_RADIAL:
            uv += self.Distortion(uv)
            return uv * f + pp
        elif self.model_id == CameraModelId.RADIAL:
            uv += self.Distortion(uv)
            return uv * f + pp
        elif self.model_id == CameraModelId.OPENCV:
            uv += self.Distortion(uv)
            return uv * ff + pp
        elif self.model_id == CameraModelId.OPENCV_FISHEYE:
            uv = self.fisheye_from_normal(uv)
            uv += self.Distortion(uv)
            return uv * ff + pp
        elif self.model_id == CameraModelId.FULL_OPENCV:
            uv += self.Distortion(uv)
            return uv * ff + pp
        elif self.model_id == CameraModelId.FOV:
            uv = self.Distortion(uv)
            return uv * f + pp
        elif self.model_id == CameraModelId.SIMPLE_RADIAL_FISHEYE:
            uv = self.fisheye_from_normal(uv)
            uv += self.Distortion(uv)
            return uv * f + pp
        elif self.model_id == CameraModelId.RADIAL_FISHEYE:
            uv = self.fisheye_from_normal(uv)
            uv += self.Distortion(uv)
            return uv * f + pp
        elif self.model_id == CameraModelId.THIN_PRISM_FISHEYE:
            uv = self.fisheye_from_normal(uv)
            uv += self.Distortion(uv)
            return uv * ff + pp
        else:
            raise NotImplementedError

class Track:
    def __init__(self, **kwargs):
        self.id = -1
        self.xyz = np.zeros(3)
        self.color = np.zeros(3)
        self.is_initialized = False
        self.observations = np.zeros(0)
        
        for key, val in kwargs.items():
            setattr(self, key, val)

class ViewGraph:
    def __init__(self):
        self.image_pairs = {} # includes: pair_id -> ImagePair
        self.num_images = 0
        self.num_pairs = 0

    def establish_adjacency_list(self):
        self.adjacency_list = {}
        for pair in self.image_pairs.values():
            if not pair.is_valid:
                continue
            if pair.image_id1 not in self.adjacency_list:
                self.adjacency_list[pair.image_id1] = set()
            self.adjacency_list[pair.image_id1].add(pair.image_id2)
            if pair.image_id2 not in self.adjacency_list:
                self.adjacency_list[pair.image_id2] = set()
            self.adjacency_list[pair.image_id2].add(pair.image_id1)

    def BFS(self, root):
        q = Queue()
        q.put(root)
        self.visited[root] = True
        component = [root]

        while not q.empty():
            current = q.get()
            for neighbor in self.adjacency_list[current]:
                if not self.visited[neighbor]:
                    q.put(neighbor)
                    self.visited[neighbor] = True
                    component.append(neighbor)

        return component

    def find_connected_component(self):
        self.connected_component = []
        self.visited = {}
        for image_id in self.adjacency_list.keys():
            self.visited[image_id] = False

        for image_id in self.adjacency_list.keys():
            if not self.visited[image_id]:
                component = self.BFS(image_id)
                self.connected_component.append(component)

    def keep_largest_connected_component(self, images):
        self.establish_adjacency_list()
        self.find_connected_component()

        max_idx = -1
        max_img = 0
        for idx, component in enumerate(self.connected_component):
            if len(component) > max_img:
                max_img = len(component)
                max_idx = idx

        if max_idx == -1:
            return False

        largest_component = self.connected_component[max_idx]
        for image in images:
            image.is_registered = image.id in largest_component

        for pair in self.image_pairs.values():
            if not images[pair.image_id1].is_registered or not images[pair.image_id2].is_registered:
                pair.is_valid = False
        return True

    def mark_connected_components(self, images):
        self.establish_adjacency_list()
        self.find_connected_component()

        cluster_num_img = []
        for comp in range(len(self.connected_component)):
            cluster_num_img.append((len(self.connected_component[comp]), comp))
        cluster_num_img.sort(key=lambda x: x[0], reverse=True)

        for image in images:
            image.cluster_id = -1

        comp = 0
        for comp in range(len(cluster_num_img)):
            for image_id in self.connected_component[cluster_num_img[comp][1]]:
                images[image_id].cluster_id = comp
        return comp + 1