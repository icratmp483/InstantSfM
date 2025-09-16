import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm

from instantsfm.utils.cost_function import pairwise_cost
from instantsfm.utils.depth_sample import sample_depth_at_pixel

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

class TorchGP():
    def __init__(self, visualizer=None, device='cuda:0'):
        super().__init__()
        self.device = device
        self.visualizer = visualizer
        
    def InitializeRandomPositions(self, cameras, images, tracks, depths=None):
        # calculate average valid depth to estimate scale of the scene
        scene_scale = 100
        if depths is not None:
            valid_depths = depths[depths > 0]
            if len(valid_depths):
                scene_scale = np.mean(valid_depths) * 4.0

        for image in images:
            image.world2cam[:3, 3] = scene_scale * np.random.uniform(-1, 1, 3)

        for track in tracks.values():
            track.xyz = scene_scale * np.random.uniform(-1, 1, 3)
            track.is_initialized = True

        if self.visualizer:
            self.visualizer.add_step(cameras, images, tracks)

    def ConvertResults(self, images):
        for image in images:
            image.world2cam[:3, 3] = -(image.world2cam[:3, :3] @ image.world2cam[:3, 3])

    def Optimize(self, cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS, depth_only=False):
        if depth_only and depths is None:
            print("Warning: No depth maps provided, skip depth-only optimization.")
            return
        
        cost_fn = pairwise_cost
        class PairwiseNonBatched(nn.Module):
            def __init__(self, camera_translations, points_3d, scales, scale_indices=None):
                super().__init__()
                self.translations = nn.Parameter(TrackingTensor(camera_translations))  # [num_cams, 3]
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                self.scales = nn.Parameter(TrackingTensor(scales))
                if scale_indices is not None:
                    all_indices = torch.arange(scales.shape[0], device=scales.device)
                    self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]
                    # self.scale_indices = scale_indices
                    # self.orig_scales = scales.clone()

            def forward(self, translations, camera_indices, point_indices, is_calibrated):
                camera_translations = self.translations
                points_3d = self.points_3d
                loss = cost_fn(points_3d[point_indices], camera_translations[camera_indices], self.scales, translations, is_calibrated[camera_indices])
                # add L1 loss between original scales and optimized scales
                '''if hasattr(self, 'scale_indices') and len(self.scale_indices) > 0:
                    l1_loss = self.scales[self.scale_indices] - self.orig_scales[self.scale_indices]
                    loss[self.scale_indices, 3] = l1_loss.squeeze(1) * 0.001 # GLOBAL_POSITIONER_OPTIONS['depth_l1_loss_weight']'''
                return loss
            
        class PairwiseNonBatchedDepthOnly(nn.Module):
            def __init__(self, camera_translations, points_3d):
                super().__init__()
                self.translations = nn.Parameter(TrackingTensor(camera_translations))  # [num_cams, 3]
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]

            def forward(self, translations, camera_indices, point_indices, is_calibrated, scales):
                camera_translations = self.translations
                points_3d = self.points_3d
                loss = cost_fn(points_3d[point_indices], camera_translations[camera_indices], scales, translations, is_calibrated[camera_indices])
                return loss

        # filter out tracks with too few observations
        for track_id in list(tracks.keys()):
            track = tracks[track_id]
            if track.observations.shape[0] < GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track']:
                del tracks[track_id]
        # filter out images that have no tracks
        image_used = np.zeros(len(images), dtype=bool)
        for track in tracks.values():
            unique_image_ids = np.unique(track.observations[:, 0])
            image_used[unique_image_ids] = True
            if all(image_used):
                break
        for image_id, image in enumerate(images):
            if not image_used[image_id]:
                image.is_registered = False
        
        image_id2idx = {}
        image_idx2id = {}
        for image_id, image in enumerate(images):
            if not image.is_registered:
                continue
            image_id2idx[image_id] = len(image_id2idx)
            image_idx2id[len(image_idx2id)] = image_id
        camera_translations_list = [(torch.tensor(img.world2cam[:3, 3])) for img in images if img.is_registered]
        camera_translations = torch.stack(camera_translations_list, dim=0).to(self.device).to(torch.float64)
        points_3d_list = [torch.tensor(track.xyz) for track in tracks.values()]
        points_3d = torch.stack(points_3d_list, dim=0).to(self.device).to(torch.float64)

        # note: camera extrinsic indices actually refer to image indices, relevant params follow the same logic
        translations_list = []
        camera_indices_list = []
        point_indices_list = []
        depth_values_list = []
        depth_availability_list = []

        for track_id, track in enumerate(tracks.values()):
            for image_id, feature_id in track.observations:
                image = images[image_id]
                if not image.is_registered:
                    continue
                if depths is not None:
                    # sample depth as scales for optimization
                    depth, available = sample_depth_at_pixel(depths[image_id], image.features[feature_id], cameras[image.cam_id].width, cameras[image.cam_id].height)
                    # only consider valid depth
                    if depth_only and not available:
                        continue
                    depth_values_list.append(1 / depth) # use inverse depth
                    depth_availability_list.append(available) # 1 for available depth
                translation = image.world2cam[:3, :3].T @ image.features_undist[feature_id]
                translations_list.append(translation)
                camera_indices_list.append(image_id2idx[image_id])
                point_indices_list.append(track_id)

        translations = torch.tensor(np.array(translations_list), dtype=torch.float64).to(self.device)
        camera_indices = torch.tensor(np.array(camera_indices_list), dtype=torch.int32).to(self.device)
        point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32).to(self.device)
        is_calibrated = torch.tensor([cameras[img.cam_id].has_prior_focal_length for img in images if img.is_registered], dtype=torch.bool).to(self.device)
        
        scale_indices = None
        if depths is None:
            scales = torch.ones(len(translations_list), 1, dtype=torch.float64, device=self.device)
        else:
            scales = torch.tensor(np.array(depth_values_list), dtype=torch.float64, device=self.device).unsqueeze(1)
            depth_availability = torch.tensor(np.array(depth_availability_list), dtype=torch.float64, device=self.device).unsqueeze(1)
            # indices for optimizer to calculate loss with valid depth scale
            scale_indices = torch.where(depth_availability == 1)[0]

        if depth_only:
            model = PairwiseNonBatchedDepthOnly(camera_translations, points_3d)
        else:
            model = PairwiseNonBatched(camera_translations, points_3d, scales, scale_indices=scale_indices)
        strategy = pp.optim.strategy.TrustRegion(radius=1e3, max=1e10, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(GLOBAL_POSITIONER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "translations": translations,
            "camera_indices": camera_indices,
            "point_indices": point_indices,
            "is_calibrated": is_calibrated,
        }
        if depth_only:
            input["scales"] = scales

        window_size = 4
        loss_history = []
        progress_bar = tqdm.trange(GLOBAL_POSITIONER_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < GLOBAL_POSITIONER_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})

            if self.visualizer:
                points_3d_np = points_3d.detach().cpu().numpy()
                camera_translations_np = camera_translations.detach().cpu().numpy()
                for track_id, track in enumerate(tracks.values()):
                    track.xyz = points_3d_np[track_id]
                for idx in range(camera_translations_np.shape[0]):
                    image = images[image_idx2id[idx]]
                    image.world2cam[:3, 3] = camera_translations_np[idx]
                self.ConvertResults(images)
                self.visualizer.add_step(cameras, images, tracks, "global_positioning")
            
        progress_bar.close()

        points_3d = points_3d.detach().cpu().numpy()
        camera_translations = camera_translations.detach().cpu().numpy()
        for track_id, track in enumerate(tracks.values()):
            track.xyz = points_3d[track_id]
        for idx in range(camera_translations.shape[0]):
            image = images[image_idx2id[idx]]
            image.world2cam[:3, 3] = camera_translations[idx]
        self.ConvertResults(images)