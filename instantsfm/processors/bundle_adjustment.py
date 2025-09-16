import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm

from instantsfm.utils.cost_function import reproject_funcs
from instantsfm.scene.defs import get_camera_model_info

# used by torch LM
import torch
from functools import partial
from torch import nn
from torch.func import jacrev
import pypose as pp
from pypose.optim.kernel import Huber
from pyro_slam.utils.pysolvers import PCG, cuSolverSP
from pyro_slam.sparse.py_ops import diagonal_op_
from pyro_slam.optim import LM
from pyro_slam.autograd.function import TrackingTensor, map_transform
from pyro_slam.utils.ba import rotate_quat

def update(cameras, images, tracks, track_keys,
           unique_cameras, unique_points, remaining_indices, pp_indices,
           camera_params, camera_pps, points_3d):
    # recover full params
    camera_params_full = torch.zeros((camera_params.shape[0], camera_params.shape[1] + 2), device=camera_params.device, dtype=torch.float64)
    camera_params_full[..., remaining_indices] = camera_params
    camera_params_full[..., pp_indices] = camera_pps

    points_3d = points_3d.detach().cpu().numpy()
    pose_matices = pp.SE3(camera_params_full[..., :7]).matrix().cpu().numpy()
    camera_params_full = camera_params_full.detach().cpu().numpy()
    for i, original_idx in enumerate(unique_points.tolist()):
        tracks[track_keys[original_idx]].xyz = points_3d[i]
    for i, image_id in enumerate(unique_cameras.tolist()):
        params = camera_params_full[i]
        image = images[image_id]
        image.world2cam = pose_matices[i]
        cam = cameras[image.cam_id]
        cam.set_params(params[7:])

class TorchBA():
    def __init__(self, visualizer=None, device="cuda:0"):
        super().__init__()
        self.device = device
        self.visualizer = visualizer

    def Solve(self, cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS):
        self.camera_model = cameras[0].model_id # assume all cameras are under the same model
        self.camera_model_info = get_camera_model_info(self.camera_model)
        try:
            cost_fn = reproject_funcs[self.camera_model.value]
        except:
            raise NotImplementedError("Unsupported camera model")
        class ReprojNonBatched(nn.Module):
            def __init__(self, camera_params, points_3d):
                super().__init__()
                self.pose = nn.Parameter(TrackingTensor(camera_params))  # [num_cams, 7 + x], x is the number of intrinsics (excluding pp)
                self.pose.requires_grad_(BUNDLE_ADJUSTER_OPTIONS['optimize_poses'])
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                self.pose.trim_SE3_grad = True

            def forward(self, points_2d, camera_indices, point_indices, camera_pps):
                camera_params = self.pose
                points_3d = self.points_3d
                points_proj = cost_fn(points_3d[point_indices], camera_params[camera_indices], camera_pps[camera_indices])
                loss = points_proj - points_2d
                return loss

        track_keys = list(tracks.keys())
        track_lengths = np.array([len(tracks[track_id].observations) for track_id in track_keys])
        is_track_valid = track_lengths >= BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track']

        image_registered = np.array([img.is_registered for img in images], dtype=bool)
        camera_params_list = [torch.cat([(pp.mat2SE3(img.world2cam) if img.is_registered else pp.identity_SE3()).tensor(), 
                                         torch.tensor(cameras[img.cam_id].params)]) for img in images]
        camera_params = torch.stack(camera_params_list, dim=0).to(self.device).to(torch.float64)

        # because principal point is not optimized, remove it from camera params
        pp_indices = torch.tensor(self.camera_model_info['pp'], device=self.device) + 7 # add 7 for translation and rotation
        camera_pps = camera_params[..., pp_indices]
        all_indices = torch.arange(camera_params.shape[1], device=self.device)
        remaining_indices = torch.tensor([i for i in all_indices if i not in pp_indices], device=self.device)
        camera_params = camera_params[..., remaining_indices]
        
        points_3d = np.stack([track.xyz for track in tracks.values()], axis=0)
        points_3d = torch.tensor(points_3d, device=self.device, dtype=torch.float64)

        points_2d_list = []
        camera_indices_list = []
        point_indices_list = []
        for track_id in is_track_valid.nonzero()[0]:
            for image_id, feature_id in tracks[track_keys[track_id]].observations:
                if not image_registered[image_id]:
                    continue
                image = images[image_id]
                point2D = image.features[feature_id]
                points_2d_list.append(point2D)
                camera_indices_list.append(image_id)
                point_indices_list.append(track_id)

        points_2d = torch.tensor(np.array(points_2d_list), dtype=torch.float64, device=self.device)
        camera_indices = torch.tensor(np.array(camera_indices_list), dtype=torch.int32, device=self.device)
        point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=self.device)      

        points_proj = rotate_quat(points_3d[point_indices], camera_params[camera_indices][..., :7])
        valid_observation = points_proj[..., 2] > 0.1  # TODO: should be proportional to the focal length 

        points_2d = points_2d[valid_observation]
        camera_indices = camera_indices[valid_observation]
        point_indices = point_indices[valid_observation]
        unique_cameras, camera_indices_ = torch.unique(camera_indices, sorted=True, return_inverse=True)
        unique_points, point_indices_ = torch.unique(point_indices, sorted=True, return_inverse=True)

        camera_pps = camera_pps[unique_cameras]
        camera_params = camera_params[unique_cameras]
        points_3d = points_3d[unique_points]

        model = ReprojNonBatched(camera_params, points_3d)
        strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "points_2d": points_2d,
            "camera_indices": camera_indices_,
            "point_indices": point_indices_,
            "camera_pps": camera_pps,
        }

        window_size = 4
        loss_history = []
        progress_bar = tqdm.trange(BUNDLE_ADJUSTER_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < BUNDLE_ADJUSTER_OPTIONS['function_tolerance']:
                    break
                if loss_history[-1] == loss_history[-2]: # no improvement likely because linear solver failed
                    break
            progress_bar.set_postfix({"loss": loss.item()})

            if self.visualizer:
                update(cameras, images, tracks, track_keys,
                       unique_cameras, unique_points, remaining_indices, pp_indices,
                       camera_params, camera_pps, points_3d)
                self.visualizer.add_step(cameras, images, tracks, "bundle_adjustment")
            
        progress_bar.close()
        
        update(cameras, images, tracks, track_keys,
               unique_cameras, unique_points, remaining_indices, pp_indices,
               camera_params, camera_pps, points_3d)
