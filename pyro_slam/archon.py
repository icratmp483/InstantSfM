# %%
import logging
import numpy as np
import torch

from pypose.function.geometry import homo2cart

# set random seed
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

with open('archon_txt/points.txt', 'r') as f:
    str_points = f.read()
    points = np.array(eval(str_points))

with open('archon_txt/normals.txt', 'r') as f:
    str_normals = f.read()
    normals = np.array(eval(str_normals))

with open('archon_txt/faceVertexCounts.txt', 'r') as f:
    str_faceVertexCounts = f.read()
    faceVertexCounts = np.array(eval(str_faceVertexCounts))

with open('archon_txt/faceVertexIndices.txt', 'r') as f:
    str_faceVertexIndices = f.read()
    faceVertexIndices = np.array(eval(str_faceVertexIndices))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def fibonacci_sphere(samples=1000):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def generate_se3_elements(samples=1000, radius=2):
    points = fibonacci_sphere(samples)
    se3_elements = []

    for point in points:
        # Create a rotation matrix that aligns the z-axis with the point
        z_axis = np.array([0, 0, 1])
        rotation_vector = np.cross(z_axis, point)
        rotation_angle = np.arccos(np.dot(z_axis, point) / np.linalg.norm(point))
        rotation = R.from_rotvec(rotation_vector * rotation_angle).as_matrix()

        # Create the SE(3) transformation matrix
        se3 = np.eye(4)
        se3[:3, :3] = rotation
        se3[:3, 3] = point * radius

        se3_elements.append(se3)

    return se3_elements

def visualize_se3_elements(se3_elements):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for se3 in se3_elements:
        point = se3[:3, 3]
        direction = se3[:3, 2]  # z-axis direction

        ax.quiver(point[0], point[1], point[2], direction[0], direction[1], direction[2], length=0.1, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SE(3) Elements on the Surface of a Sphere')
    plt.show()

torch.set_default_dtype(torch.float64)
# Example usage
se3_elements = generate_se3_elements(samples=11, radius=3*(points.max() - points.min()))
# example tartan air intrinsics
fx, fy, cx, cy = 320., 320., 320., 240.
K = torch.Tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
import pypose as pp
SE3_lt = pp.mat2SE3(np.array(se3_elements))
# add noise to camera
SE3_lt = SE3_lt.Retr( pp.se3(torch.randn(SE3_lt.shape[0], 6) * 0.00005 ))
points = torch.from_numpy(points)
pixels_gt = pp.function.point2pixel(points.unsqueeze(0), K.unsqueeze(0), SE3_lt) # cam x n_points x 2

noise = torch.randn(*points.shape) * 0.0005
points_org = points.clone()
points = points + noise


# %%
import torch
import warnings
from typing import Callable, List, Optional
from torch.library import Library
from tqdm import tqdm
from torchvision.transforms import Compose

# diag is already mature 
from pyro_slam.sparse.py_ops import *
from pyro_slam.sparse.bsr_cuda import *
from pyro_slam.sparse.solve import *

import torch.nn as nn
import pypose as pp

# %% [markdown]
# # Bundle Adjustment Example using SparsePyBA and the BAL dataset
# 
# ```
# The dataset is from the following paper:  
# Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.  
# Bundle adjustment in the large.  
# In European Conference on Computer Vision (ECCV), 2010.  
# ```
# 
# Link to the dataset: https://grail.cs.washington.edu/projects/bal/

# %% [markdown]
# # Fetch data

# %%
from datapipes.bal_loader import get_problem, read_bal_data

TARGET_DATASET = "ladybug"
TARGET_PROBLEM = "problem-49-7776-pre"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_PROBLEM = "problem-1695-155710-pre"  
# TARGET_PROBLEM = "problem-969-105826-pre"


# TARGET_DATASET = "trafalgar"
# TARGET_PROBLEM = "problem-21-11315-pre"

DEVICE = 'cuda' # change device to CPU if needed
DTYPE = torch.float64
USE_QUATERNIONS = True
OPTIMIZE_INTRINSICS = False
# dataset = read_bal_data(file_name='Data/dubrovnik-3-7-pre.txt', use_quat=USE_QUATERNIONS)
dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=USE_QUATERNIONS)

if OPTIMIZE_INTRINSICS:
    NUM_CAMERA_PARAMS = 10 if USE_QUATERNIONS else 9
else:
    NUM_CAMERA_PARAMS = 7 if USE_QUATERNIONS else 6

print(f'Fetched {TARGET_PROBLEM} from {TARGET_DATASET}')

torch.set_default_dtype(DTYPE)

# %% [markdown]
# # Declare helper functions

# %%
from pyro_slam.utils.ba import construct_sbt, cuSolverSP, rotate_euler, rotate_quat

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_QUATERNIONS:
        points_proj = rotate_quat(points, camera_params[..., :7])
    else:
        points_proj = rotate_euler(points, camera_params[..., 3:6])
        points_proj = points_proj + camera_params[..., :3]
    pixels = points_proj
    pixels = pixels @ K.to(pixels.device).T
    pixels = homo2cart(pixels)
    return pixels

# sparse version
class ReprojNonBatched(nn.Module):
    def __init__(self, camera_params, points_3d):
        super().__init__()
        self.pose = nn.Parameter(camera_params)
        self.points_3d = nn.Parameter(points_3d)

    def forward(self, points_2d, intr, camera_indices, point_indices):
        camera_params = self.pose
        points_3d = self.points_3d
        if intr is not None:
            camera_params = torch.cat([camera_params, intr], dim=-1)
        points_proj = project(points_3d[point_indices], camera_params[camera_indices])
        loss = points_proj - points_2d
        return loss

from functools import partial

def least_square_error(camera_params, points_3d, camera_indices, point_indices, points_2d, intr=None):
    model = ReprojNonBatched(camera_params, points_3d)
    loss = model(points_2d, intr, camera_indices, point_indices)
    return torch.sum(loss**2) / 2

# %% [markdown]
# 

# %%
from torch.func import jacrev



def modjacrev_vmap(model, input, argnums=0, *, has_aux=False):
    params = dict(model.named_parameters())

    cameras_num = params['model.pose'].shape[0]
    points_3d_num = params['model.points_3d'].shape[0]
    # need to align the indices with the parameters
    camera_indices = input['camera_indices']
    point_indices = input['point_indices']
    params['model.pose'] = params['model.pose'][camera_indices] # index using camera indices
    params['model.points_3d'] = params['model.points_3d'][point_indices] # index using point indices
    jac_points_3d, jac_pose = torch.vmap(jacrev(project, argnums=(0, 1), has_aux=has_aux))(params['model.points_3d'], params['model.pose'])
    if USE_QUATERNIONS: 
        useful_idx = [0,1,2,3,4,5,7,8,9] if OPTIMIZE_INTRINSICS else [0,1,2,3,4,5,]
        jac_pose = jac_pose[..., useful_idx] # remove the 4th element of the quaternion
                                                    # because original is [qx, qy, qz, qw, tx, ty, tz], but always dqw = 0
    return [construct_sbt(jac_pose, cameras_num, camera_indices), construct_sbt(jac_points_3d, points_3d_num, point_indices)]

# %% [markdown]
# # Run optimization

# %%



# %%
class LM(pp.optim.LevenbergMarquardt):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input))
            J = modjacrev_vmap(self.model, input)

            # params = dict(self.model.named_parameters())
            # params_values = tuple(params.values())
            # J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
            # for i in range(len(R)):
            #     R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
            #         else self.corrector[i](R = R[i], J = J[i])
            R = R[0]
            J = torch.cat([j.to_sparse_coo() for j in J], dim=-1)

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            J_T = J.T @ weight if weight is not None else J.T
            A, self.reject_count = J_T @ J, 0
            # A = A.to_sparse_bsr(blocksize=(1,1))
            A = A.to_sparse_csr()
            diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            while self.last <= self.loss:
                diagonal_op_(A, op=partial(torch.mul, other=1+pg['damping']))
                try:
                    D = self.solver(A = A, b = -J_T @ R.view(-1, 1))
                    D = D[:, None]
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject: # reject step
                    self.update_parameter(params = pg['params'], step = -D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss
    def update_parameter(self, params, step):
        numels = []
        for i, p in enumerate(params):
            if p.requires_grad:
                if i == 0:
                    numels.append(p.shape[0] * (9 if OPTIMIZE_INTRINSICS else 6))
                else:
                    numels.append(p.numel())
        steps = step.split(numels)
        for i, (p, d) in enumerate(zip(params, steps)):
            if p.requires_grad:
                if i == 0:
                    # continue
                    if USE_QUATERNIONS:
                        p[..., :7] = pp.SE3(p[..., :7]).add_(pp.se3(d.view(p.shape[0], -1)[..., :6]))
                        if OPTIMIZE_INTRINSICS: p[:, 7:] += d.view(p.shape[0], -1)[:, 6:]
                        continue
                p.add_(d.view(p.shape))


class Schur(LM):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = self.model(input, target)
            J = modjacrev_vmap(self.model, input)

            R = R[0]
            J[0] = J[0]
            J[1] = J[1]

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            # torch.cuda.nvtx.range_push("JTJc")
            U = J[0].mT @ J[0]
            # torch.cuda.nvtx.range_pop()
            # J0D = J[0].to_dense()
            # UD = U.to_dense()
            # torch.testing.assert_close(UD, J0D.mT @ J0D)
            # del J0D
            # del UD
            # torch.cuda.nvtx.range_push("JTJp")
            V = J[1].mT @ J[1]
            # torch.cuda.nvtx.range_pop()
            # J1D = J[1].to_dense()
            # VD = V.to_dense()
            # torch.testing.assert_close(VD, J1D.mT @ J1D)
            # del J1D
            # del VD
            
            # torch.cuda.nvtx.range_push("Clamp")
            diagonal_op_(U, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(V, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            # torch.cuda.nvtx.range_pop()

            while self.last <= self.loss:
                damping = pg['damping']
                R = R.reshape(-1)
                
                # torch.cuda.nvtx.range_push("Damp")
                diagonal_op_(U, op=partial(torch.add, other=(torch.diagonal(U).pow(2)) * damping))
                diagonal_op_(V, op=partial(torch.add, other=(torch.diagonal(V).pow(2)) * damping))
                # torch.cuda.nvtx.range_pop()

                # torch.cuda.nvtx.range_push("W")
                W = J[0].mT @ J[1]
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("Ic")
                Ic = -J[0].mT.to_sparse_coo().to_sparse_csr() @ R
                Ip = -J[1].mT.to_sparse_coo().to_sparse_csr() @ R
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("Inv")
                V_i = inv_op(V)
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("WVi")
                WV_i = W @ V_i
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("rhs1")
                rhs = Ic - WV_i.to_sparse_coo().to_sparse_csr() @ Ip  
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("lhs1")
                lhs = add_op(U, (-WV_i @ W.mT))  # this matrix is NOT symetric
                # torch.cuda.nvtx.range_pop()

                # torch.cuda.nvtx.range_push("Solve C")
                try:
                    D_c = self.solver(A = lhs.to_sparse_coo().to_sparse_csr(), b = rhs)
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                # torch.cuda.nvtx.range_pop()
                
                # torch.cuda.nvtx.range_push("rhs2")
                rhs = Ip - W.mT.to_sparse_coo() @ D_c
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("solve2")
                lhs = V
                D_p = self.solver(A = lhs.to_sparse_coo().to_sparse_csr(), b = rhs)
                # torch.cuda.nvtx.range_pop()
                # torch.cuda.nvtx.range_push("Update")
                D = torch.cat([D_c, D_p])
                self.update_parameter(pg['params'], D)
                # torch.cuda.nvtx.range_pop()
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                # torch.cuda.nvtx.range_push("Strategy")
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=[D_c, D_p], R=R.view(-1, 1))
                # torch.cuda.nvtx.range_pop()
                if self.last < self.loss and self.reject_count < self.reject: # reject step
                    self.update_parameter(params = pg['params'], step = -D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss
    
    # def _update_parameter(self, params, step):
        
    #     V, Ip, W = self.model.cur['V'], self.model.cur['Ip'], self.model.cur['W']
    #     rhs = Ip - W.mT.to_sparse_coo() @ step
    #     lhs = V
    #     D_p = self.solver(A = lhs, b = rhs)
    #     params[1] += D_p.view_as(params[1])
    #     return step, D_p

# %%
camera_params_other = torch.Tensor([fx, cx, cy]).to(DEVICE).unsqueeze(0)
camera_params_other = camera_params_other.repeat(pixels_gt.shape[0], 1)
pixels_gt = pixels_gt.to(DEVICE)
camera_indices = []
for i in range(pixels_gt.shape[0]):
    camera_indices.append(torch.ones(pixels_gt.shape[1], dtype=torch.int64) * i)
camera_indices = torch.cat(camera_indices).contiguous().flatten()
point_indices = torch.arange(pixels_gt.shape[1], dtype=torch.int64).repeat(pixels_gt.shape[0]).contiguous()
input = {"points_2d": pixels_gt.flatten(0, 1),
         "intr": camera_params_other,
         "camera_indices": camera_indices.to(DEVICE),
         "point_indices": point_indices.to(DEVICE)}

# gtsam coord
# trimmed_dataset['camera_params'][:, :7] = Compose([pp.SE3, openGL2gtsam])(trimmed_dataset['camera_params'][:, :7])
# trimmed_dataset['points_2d'][:, 1] = -trimmed_dataset['points_2d'][:, 1]

model_non_batched = ReprojNonBatched(torch.Tensor(SE3_lt).clone().to(DEVICE),
                                     points.clone().to(DEVICE))

model_non_batched = model_non_batched.to(DEVICE)

# strategy_sparse = pp.optim.strategy.TrustRegion()
strategy_sparse = pp.optim.strategy.Adaptive(damping=0.0001, min=1.5e-9)
# sparse_solver = PCG(tol=1e-3, maxiter=10000)
# sparse_solver = SciPySpSolver()
sparse_solver = cuSolverSP()
optimizer_sparse = LM(model_non_batched, strategy=strategy_sparse, solver=sparse_solver, reject=30)

# least_square_error(camera_params, points_3d, camera_indices, point_indices, points_2d, intr=None)

print('Starting loss:', least_square_error(model_non_batched.pose, model_non_batched.points_3d, camera_indices, point_indices, pixels_gt.flatten(0, 1), intr=camera_params_other).item())
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for idx in range(1000):
    loss = optimizer_sparse.step(input)

torch.cuda.synchronize()
end.record()
print('Time', start.elapsed_time(end) / 1000)

print('Loss:', least_square_error(model_non_batched.pose, model_non_batched.points_3d, camera_indices, point_indices, pixels_gt.flatten(0, 1), intr=camera_params_other).item())
print('Ending loss:', least_square_error(model_non_batched.pose, model_non_batched.points_3d, camera_indices, point_indices, pixels_gt.flatten(0, 1), intr=camera_params_other).item())

# %%



