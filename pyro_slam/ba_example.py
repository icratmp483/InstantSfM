import math
from time import perf_counter
import torch

from pyro_slam.sparse.py_ops import *
from pyro_slam.sparse.solve import *
from pyro_slam.utils.ba import rotate_euler, rotate_quat
from functools import partial

import torch.nn as nn
import pypose as pp

from datapipes.bal_loader import get_problem, read_bal_data
from pyro_slam.utils.pysolvers import PCG, cuSolverSP
from pyro_slam.autograd.function import TrackingTensor, map_transform

# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-49-7776-pre"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_PROBLEM = "problem-1695-155710-pre"  
# TARGET_PROBLEM = "problem-969-105826-pre"

TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"

# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

DEVICE = 'cuda'
DTYPE = torch.float64
USE_QUATERNIONS = True
OPTIMIZE_INTRINSICS = True

file_name = f'{TARGET_DATASET}.{TARGET_PROBLEM}'
# file_name='1dsfm_bal/Alamo.txt'
# file_name='1dsfm_bal/Union_Square.txt'
# file_name='1dsfm_bal/Gendarmenmarkt.txt'
# file_name='1dsfm_bal/NYC_Library.txt'
# file_name='1dsfm_bal/Montreal_Notre_Dame.txt'
# file_name='1dsfm_bal/Yorkminster.txt'
# file_name='1dsfm_bal/Roman_Forum.txt'
# file_name='1dsfm_bal/Vienna_Cathedral.txt'
# file_name='1dsfm_bal/Madrid_Metropolis.txt'
# file_name='1dsfm_bal/Piccadilly.txt'
# file_name='1dsfm_bal/Tower_of_London.txt'
# file_name='1dsfm_bal/Trafalgar.txt'

dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=USE_QUATERNIONS)
# dataset = read_bal_data(file_name, use_quat=USE_QUATERNIONS)

if OPTIMIZE_INTRINSICS:
    NUM_CAMERA_PARAMS = 10 if USE_QUATERNIONS else 9
else:
    NUM_CAMERA_PARAMS = 7 if USE_QUATERNIONS else 6

print(f'Fetched {TARGET_PROBLEM} from {TARGET_DATASET}')

trimmed_dataset = dataset
trimmed_dataset = {k: v.to(DEVICE) for k, v in trimmed_dataset.items() if type(v) == torch.Tensor}

def convert_to(type):
    for k, v in trimmed_dataset.items():
        if 'index' not in k:
            trimmed_dataset[k] = v.to(type)

convert_to(DTYPE)
torch.set_default_dtype(DTYPE)


@map_transform
def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_QUATERNIONS:
        points_proj = rotate_quat(points, camera_params[..., :7])
    else:
        points_proj = rotate_euler(points, camera_params[..., 3:6])
        points_proj = points_proj + camera_params[..., :3]
    points_proj = -points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    f = camera_params[..., -3].unsqueeze(-1)
    k1 = camera_params[..., -2].unsqueeze(-1)
    k2 = camera_params[..., -1].unsqueeze(-1)
    
    n = torch.sum(points_proj**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    points_proj = points_proj * r * f

    return points_proj


class ReprojNonBatched(nn.Module):
    def __init__(self, camera_params, points_3d):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.pose.trim_SE3_grad = True

    def forward(self, points_2d, camera_indices, point_indices):
        camera_params = self.pose
        points_3d = self.points_3d

        points_proj = project(points_3d[point_indices], camera_params[camera_indices])
        loss = points_proj - points_2d
        return loss


def least_square_error(camera_params, points_3d, camera_indices, point_indices, points_2d):
    model = ReprojNonBatched(camera_params, points_3d)
    loss = model(points_2d, camera_indices, point_indices)
    return torch.sum(loss**2, dim=-1).mean()
    return torch.sum(loss**2) / 2



input = {
    "points_2d": trimmed_dataset['points_2d'],
    "camera_indices": trimmed_dataset['camera_index_of_observations'],
    "point_indices": trimmed_dataset['point_index_of_observations']
}

# gtsam coord
# trimmed_dataset['camera_params'][:, :7] = Compose([pp.SE3, openGL2gtsam])(trimmed_dataset['camera_params'][:, :7])
# trimmed_dataset['points_2d'][:, 1] = -trimmed_dataset['points_2d'][:, 1]

model = ReprojNonBatched(
    trimmed_dataset['camera_params'][:, :NUM_CAMERA_PARAMS].clone(),
    trimmed_dataset['points_3d'].clone()
)

model = model.to(DEVICE)
from pyro_slam.optim import LM
strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
# strategy = pp.optim.strategy.Adaptive(damping=0.0001, min=1.5e-6)
# solver = PCG(tol=1e-3, maxiter=10000)
# solver = SciPySpSolver()
# solver = PCG(tol=1e-4)
solver = cuSolverSP()
optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

print('Starting loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d']
).item())
start = perf_counter()
for idx in range(7):
    # to dump the pcl, uncomment:
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(model_non_batched.points_3d.detach().cpu().numpy())
    # o3d.io.write_point_cloud(f'./{file_name}_iter{idx}_points_3d.ply', pcd)
    loss = optimizer.step(input)

torch.cuda.synchronize()
end = perf_counter()
print('Time', end - start)

print('Loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
).item())
print('Ending loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
).item())
