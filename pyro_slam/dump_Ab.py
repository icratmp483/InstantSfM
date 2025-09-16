# %%
import os
import torch
import warnings
from typing import Callable, List, Optional
from torch.library import Library
from tqdm import tqdm
from torchvision.transforms import Compose

# diag is already mature 
from pyro_slam.sparse.py_ops import *
from pyro_slam.sparse.bsr import *


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
import sys

TARGET_DATASET = sys.argv[1]
TARGET_PROBLEM = sys.argv[2]
DUMP_DIR = 'dump_Ab'

# TARGET_DATASET = "trafalgar"
# TARGET_PROBLEM = "problem-21-11315-pre"

DEVICE = 'cpu' # change device to CPU if needed
DTYPE = torch.float64
USE_QUATERNIONS = True
OPTIMIZE_INTRINSICS = True
# dataset = read_bal_data(file_name='Data/dubrovnik-3-7-pre.txt', use_quat=USE_QUATERNIONS)
dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=USE_QUATERNIONS)

if OPTIMIZE_INTRINSICS:
  NUM_CAMERA_PARAMS = 10 if USE_QUATERNIONS else 9
else:
  NUM_CAMERA_PARAMS = 7 if USE_QUATERNIONS else 6

print(f'Fetched {TARGET_PROBLEM} from {TARGET_DATASET}')

import torch
import torch.nn as nn
import pypose as pp

def trim_dataset(dataset, max_pixels=None):
  trimmed_dataset = dict()
  if max_pixels is None:
    max_pixels = dataset['points_2d'].shape[0]
  trimmed_dataset['points_2d'] = dataset['points_2d'][:max_pixels]
  trimmed_dataset['point_index_of_observations'] = dataset['point_index_of_observations'][:max_pixels]
  trimmed_dataset['camera_index_of_observations'] = dataset['camera_index_of_observations'][:max_pixels]
  # other fields are not changed
  trimmed_dataset['camera_params'] = dataset['camera_params']
  trimmed_dataset['points_3d'] = dataset['points_3d']

  for k in trimmed_dataset.keys():
    if not isinstance(trimmed_dataset[k], torch.Tensor):
      trimmed_dataset[k] = torch.from_numpy(trimmed_dataset[k])
    trimmed_dataset[k] = trimmed_dataset[k].to(DEVICE)
  return trimmed_dataset

trimmed_dataset = trim_dataset(dataset)

def convert_to(type):
  for k, v in trimmed_dataset.items():
    if 'index' not in k:
      trimmed_dataset[k] = v.to(type)


convert_to(DTYPE)
torch.set_default_dtype(DTYPE)

# %% [markdown]
# # Declare helper functions

# %%
import torch

def rotate_euler(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = torch.norm(rot_vecs, dim=1, keepdim=True)
    v = torch.nan_to_num(rot_vecs / theta)
    dot = torch.sum(points * v, dim=1, keepdim=True)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return cos_theta * points + sin_theta * torch.cross(v, points, dim=1) + dot * (1 - cos_theta) * v

def rotate_quat(points, rot_vecs):
    rot_vecs = pp.SE3(rot_vecs).Inv()
    return rot_vecs.Act(points)

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_QUATERNIONS:
        points_proj = rotate_quat(points, camera_params[:, [4,5,6,0,1,2,3]])
        # points_proj = points_proj
        points_proj = points_proj[:, :2] / points_proj[:, 2].unsqueeze(-1)  # add dimension for broadcasting
        f = camera_params[:, 7].unsqueeze(-1)
        k1 = camera_params[:, 8].unsqueeze(-1)
        k2 = camera_params[:, 9].unsqueeze(-1)
    else:
        points_proj = rotate_euler(points, camera_params[:, :3])
        points_proj = points_proj + camera_params[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2].unsqueeze(-1)  # add dimension for broadcasting
        f = camera_params[:, 6].unsqueeze(-1)
        k1 = camera_params[:, 7].unsqueeze(-1)
        k2 = camera_params[:, 8].unsqueeze(-1)
    
    n = torch.sum(points_proj**2, axis=1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    points_proj = points_proj * r * f  # broadcasting will take care of the shape

    return points_proj

from torch.autograd import Function
class QuatProjExplicitDiff(Function):
    @staticmethod
    def forward(points, camera_params):
        return project(points, camera_params)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        Y = output
        ctx.save_for_backward(Y, )
        return

    @staticmethod
    def backward(grad_output):
        return

def reprojerr(camera_params, points_3d, camera_indices, point_indices, points_2d, intr=None):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    if intr is not None:
        camera_params = torch.cat([camera_params, intr], dim=1)
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    loss = points_proj - points_2d
    return loss

def reprojerr_vmap(camera_params, point_3d, point_2d, intr=None):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = camera_params.unsqueeze(0)
    if intr is not None:
        intr = intr.unsqueeze(0)
        camera_params = torch.cat([camera_params, intr], dim=1)
    points_3d = point_3d.unsqueeze(0)
    points_proj = project(points_3d, camera_params).squeeze(0) # remove the batch dimension
    
    loss = points_proj - point_2d
    return loss

# def reprojerr(pose, points, intrinsics, distortions, pixels, camera_index, point_index):
#   points = points[point_index, None] # [1000, 1, 3]
#   pose = pose[camera_index] # [1000, 7]
#   points = pose.unsqueeze(-2) @ points
#   points = points.squeeze(-2)

#   # perspective division
#   points_proj = -points[:, :2] / points[:, -1:]

#   # convert to pixel coordinates
#   intrinsics = intrinsics[camera_index]
#   distortions = distortions[camera_index]
#   f = intrinsics[:, 0, 0]
#   k1 = distortions[:, 0]
#   k2 = distortions[:, 1]
#   n = torch.sum(points_proj**2, dim=-1)
#   r = 1.0 + k1 * n + k2 * n**2
#   img_repj = f[:, None] * r[:, None] * points_proj

#   # calculate the reprojection error
#   loss = (img_repj - pixels)

#   return loss

# def reprojerr_vmap(pose, point, intrinsic, distortion, pixel):
#   # reprojerr_vmap is not batched, it operates on a single 3D point and camera
#   pose = pp.LieTensor(pose, ltype=pp.SE3_type) # pose will lose its ltype through vmap, temporary fix
#   point = pose.unsqueeze(-2) @ point
#   point = point.squeeze(-2)

#   # perspective division
#   point_proj = -point[:2] / point[-1:]

#   # convert to pixel coordinates
#   f = intrinsic[0, 0]
#   k1 = distortion[0]
#   k2 = distortion[1]
#   n = torch.sum(point_proj**2, dim=-1)
#   r = 1.0 + k1 * n + k2 * n**2
#   img_repj = f * r * point_proj

#   # calculate the reprojection error
#   loss = (img_repj - pixel)

#   return loss

def reprojerr_gen(*args):
    # args: pose_param, points_3d_param
    # input (batch): points_2d, pose_other, camera_indices, point_indices
    # backward (non-batch, vmap): points_2d, pose_other
    pose = args[0]
    points_3d = args[1]
    points_2d = args[2]

    if pose.ndim == 2:
        # batch mode
        # concat params [pose, pose_other, points_3d]
        # reprojerr(camera_params, points_3d, camera_indices, point_indices, points_2d, intr=None)
        pose_other = args[3]
        camera_indices = args[4]
        point_indices = args[5]
        return reprojerr(pose, points_3d, camera_indices, point_indices, points_2d, intr=pose_other)
    else:
        # vmap mode
        # reprojerr_vmap(camera_params, point_3d, point_2d, intr=None)
        pose_other = args[3] if len(args) > 3 else None # pose_other could be non-existent in vmap mode
        return reprojerr_vmap(pose, points_3d, points_2d, intr=pose_other)

# sparse version
class ReprojNonBatched(nn.Module):
    def __init__(self, camera_params, points_3d):
        super().__init__()
        self.pose = nn.Parameter(camera_params)
        self.points_3d = nn.Parameter(points_3d)

    def forward(self, *args):
        # concatenate the parameters with args
        return reprojerr_gen(self.pose, self.points_3d, *args)

def least_square_error(camera_params, points_3d, camera_indices, point_indices, points_2d, intr=None):
    loss = reprojerr(camera_params, points_3d, camera_indices, point_indices, points_2d, intr)
    return torch.sum(loss**2) / 2

# %% [markdown]
# 

# %%
from functools import partial
from torch.func import jacrev, jacfwd, functional_call


def construct_sbt(jac_from_vmap, num, index):
    n = index.shape[0] # num 2D points
    i = torch.stack([torch.arange(n).to(index.device), index])
    block_shape = jac_from_vmap.shape[1:]
    v = jac_from_vmap # adjust dimension to accomodate for sbt constructor
    dummy_val = torch.arange(n, device=index.device, dtype=torch.int64)
    dummy_coo = torch.sparse_coo_tensor(i, dummy_val, size=(n, num), device=index.device, dtype=torch.int64)
    dummy_csc = dummy_coo.coalesce().to_sparse_csc()
    return torch.sparse_bsc_tensor(ccol_indices = dummy_csc.ccol_indices(), 
                                   row_indices=dummy_csc.row_indices(),
                                   values = v[dummy_csc.values()],
                                   size = (n * block_shape[0], num * block_shape[1]),
                                   device=index.device, dtype=DTYPE)

def construct_sbt_points_3d(jac_from_vmap, num, index):
    n = index.shape[0] # num 2D points
    i = torch.stack([torch.arange(n).to(index.device), index])
    v = jac_from_vmap
    return pp.sbktensor(i, v, size=(n, num), device=index.device, dtype=DTYPE)

def modjacrev_vmap(model, input, argnums=0, *, has_aux=False):
    params = dict(model.named_parameters())
    func = partial(functional_call, model)
    cameras_num = params['model.pose'].shape[0]
    points_3d_num = params['model.points_3d'].shape[0]
    # need to align the indices with the parameters
    camera_indices = input[-2]
    point_indices = input[-1]
    params['model.pose'] = params['model.pose'][camera_indices] # index using camera indices
    params['model.points_3d'] = params['model.points_3d'][point_indices] # index using point indices
    points_2d = input[0]
    camera_params_other = input[1][camera_indices] if input[1] is not None else None
    vmap_input = [points_2d] if camera_params_other is None else [points_2d, camera_params_other]
    jac_dict = torch.vmap(jacrev(func, argnums=argnums, has_aux=has_aux))(params, vmap_input)
    jac_pose = jac_dict[0]['model.pose']
    if USE_QUATERNIONS: 
        useful_idx = [0,1,2,4,5,6,7,8,9] if OPTIMIZE_INTRINSICS else [0,1,2,4,5,6]
        jac_pose = jac_pose[..., useful_idx] # remove the 4th element of the quaternion
                                                    # because original is [qx, qy, qz, qw, tx, ty, tz], but always dqw = 0
    jac_points_3d = jac_dict[0]['model.points_3d'].squeeze(-2)
    #jac_intrinsics = jac_dict[0]['model.intrinsics']
    #jac_distortions = jac_dict[0]['model.distortions']
    return [construct_sbt(jac_pose, cameras_num, camera_indices), construct_sbt(jac_points_3d, points_3d_num, point_indices)]

# %% [markdown]
# # Run optimization

# %%
import time
from typing import Optional

from torch import Tensor


class SchurModel(pp.optim.optimizer.RobustModel):
    def __init__(self, model, kernel=None, auto=False):
        super().__init__(model, kernel, auto)
    
    def forward(self, input, target=None):
        return super().forward(input, target)
    
    def normalize_RWJ(self, R, weight, J, pg):
        if pg is not None:
            damping = pg['damping']
        else:
            damping = 0
        R = R[0].reshape(-1)
        start_time = time.perf_counter()
        U = J[0].mT @ J[0]
        diagonal_op_(U, op=partial(torch.add, other=(torch.diagonal(U).pow(2)) * damping))
        V = J[1].mT @ J[1]
        diagonal_op_(V, op=partial(torch.add, other=(torch.diagonal(V).pow(2)) * damping))

        diagonal_op_(U, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
        diagonal_op_(V, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))


        elapsed_time = time.perf_counter() - start_time
        print(f"bsr @ bsc {elapsed_time} seconds")
        W = J[0].mT @ J[1]
        Ic = -J[0].mT.to_sparse_coo().to_sparse_csr() @ R
        Ip = -J[1].mT.to_sparse_coo().to_sparse_csr() @ R
        # return Ic, weight, U
        # return Ip, weight, V
        V_i = inv_op(V)
        WV_i = W @ V_i
        rhs = Ic - WV_i.to_sparse_coo().to_sparse_csr() @ Ip  # multiplication result in one number nan
        lhs = add_op(U, (-WV_i @ W.mT))  # this matrix is NOT symetric
        self.cur = {
            'V': V,
            'Ip': Ip,
            'W': W,
        }
        # apply PCG
        # l_diag = lhs.diagonal()
        # l_diag[l_diag.abs() < 1e-5] = 1e-5
        # M = torch.sparse.spdiags(1 / l_diag[None].cpu(), offsets=torch.zeros(1, dtype=int), shape=lhs.shape)
        # M = M.to_sparse_bsr(blocksize=U.values().shape[-2:]).to(DEVICE)
        # rhs = M @ rhs
        # lhs = M @ lhs.to_sparse_bsc(blocksize=lhs.values().shape[-2:])
        return rhs, weight, lhs
    
class TrustRegion(pp.optim.strategy.TrustRegion):
    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        J = [i.to_sparse_coo() for i in J]
        JD = None
        for i in range(len(D)):
            if JD is None:
                JD = J[i] @ D[i]
            else:
                JD += J[i] @ D[i]
        JD = JD[..., None]
        quality = (last - loss) / -((JD).mT @ (2 * R.view_as(JD) + JD)).squeeze()
        pg['radius'] = 1. / pg['damping']
        if quality > pg['high']:
            pg['radius'] = pg['up'] * pg['radius']
            pg['down'] = self.down
        elif quality > pg['low']:
            pg['radius'] = pg['radius']
            pg['down'] = self.down
        else:
            pg['radius'] = pg['radius'] * pg['down']
            pg['down'] = pg['down'] * pg['factor']
        pg['down'] = max(self.min, min(pg['down'], self.max))
        pg['radius'] = max(self.min, min(pg['radius'], self.max))
        pg['damping'] = 1. / pg['radius']

class Adaptive(pp.optim.strategy.Adaptive):
    def update(self, pg, last, loss, J, D, R, *args, **kwargs):
        J = [i.to_sparse_coo() for i in J]
        JD = None
        for i in range(len(D)):
            if JD is None:
                JD = J[i] @ D[i]
            else:
                JD += J[i] @ D[i]
        JD = JD[..., None]
        quality = (last - loss) / -((JD).mT @ (2 * R.view_as(JD) + JD)).squeeze()
        if quality > pg['high']:
            pg['damping'] = pg['damping'] * pg['down']
        elif quality > pg['low']:
            pg['damping'] = pg['damping']
        else:
            pg['damping'] = pg['damping'] * pg['up']
        pg['damping'] = max(self.min, min(pg['damping'], self.max))


class Schur(pp.optim.LevenbergMarquardt):
    def __init__(self, model, solver=None, strategy=None, kernel=None, corrector=None, weight=None, reject=16, min=0.000001, max=1e+32, vectorize=True):
        super().__init__(model, solver, strategy, kernel, corrector, weight, reject, min, max, vectorize)
        self.model = SchurModel(model, kernel)
    
    def update_parameter(self, params, step):
        # param 0 will be camera parameters, the last dimension will be CAMERA_PARAMS, format: [translation, rotation, <intrinsics>]
        # param 1 will be 3D points, the last dimension will be 3
        if USE_QUATERNIONS:
            params[0][:, :7] = pp.SE3(params[0][:, :7]).Retr(pp.se3(step.view(params[0].shape[0], -1)[:, :6]))
            if OPTIMIZE_INTRINSICS: params[0][:, 7:] += step.view(params[0].shape[0], -1)[:, 6:]
        else:
            params[0] += step.view_as(params[0])
        V, Ip, W = self.model.cur['V'], self.model.cur['Ip'], self.model.cur['W']
        rhs = Ip - W.mT.to_sparse_coo() @ step
        lhs = V
        D_p = self.solver(A = lhs, b = rhs)
        params[1] += D_p.view_as(params[1])
        return step, D_p


    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))

            R_ = R
            J = modjacrev_vmap(self.model, input)
            # params = dict(self.model.named_parameters())
            # params_values = tuple(params.values())
            # J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
            # for i in range(len(R)):
            #     R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
            #         else self.corrector[i](R = R[i], J = J[i])
            self.cur_J = J

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            self.reject_count = 0
            while self.last <= self.loss:
                b, weight, A = self.model.normalize_RWJ(R, weight, J, pg)
                diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
                
                try:
                    D = self.solver(A = A, b = b)
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                step_applied = self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                self.strategy.update(pg, last=self.last, loss=self.loss, J=self.cur_J, D=step_applied, R=R_[0])
                print(self.loss, self.last, self.reject_count, pg['damping'])
                if self.last < self.loss:
                    for idx, (para_, step_applied_) in enumerate(zip(pg['params'], step_applied)):
                        if USE_QUATERNIONS and idx == 0:
                            para_[:, :7] = pp.SE3(para_[:, :7]).Retr(pp.se3(-step_applied_.view(para_.shape[0], -1)[:, :6]))
                            if OPTIMIZE_INTRINSICS: para_[:, 7:] -= step_applied_.view(para_.shape[0], -1)[:, 6:]
                            continue
                        para_ -= step_applied_.view_as(para_)
                    if self.reject_count < self.reject: # reject step
                        self.loss, self.reject_count = self.last, self.reject_count + 1
                    else:
                        break
        return self.loss

from pypose.optim.solver import CG


class PCG(CG):
    def __init__(self, maxiter=None, tol=0.00001):
        super().__init__(maxiter, tol)
    def forward(self, A: Tensor, b: Tensor, x: Tensor | None = None, M: Tensor | None = None) -> Tensor:
        lhs = A
        rhs = b
        if b.dim() == 1:
            b = b[..., None]
        l_diag = lhs.diagonal()
        l_diag[l_diag.abs() < 1e-6] = 1e-6
        M = torch.sparse.spdiags(1 / l_diag[None].cpu(), offsets=torch.zeros(1, dtype=int), shape=lhs.shape)
        M = M.to_sparse_bsr(blocksize=A.values().shape[-2:]).to(DEVICE)
        rhs = M @ rhs
        lhs = M @ lhs.to_sparse_bsc(blocksize=lhs.values().shape[-2:])

        return super().forward(lhs, rhs, x)

class SciPySpSolver(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, A, b):
        import scipy.sparse.linalg as spla
        import scipy.sparse as sp
        import numpy as np
        if A.layout != torch.sparse_csr:
            A = A.to_sparse_coo().to_sparse_csr()
        A_csr = sp.csr_matrix((A.values().cpu().numpy(), 
                                   A.col_indices().cpu().numpy(),
                                   A.crow_indices().cpu().numpy()),
                                  shape=A.shape)
        b = b.cpu().numpy()
        x = spla.spsolve(A_csr, b, use_umfpack=False)
        assert not np.isnan(x).any()
        # a_err = np.linalg.norm(A_csr @ x - b)
        # r_err = a_err / np.linalg.norm(b)
        # print(f"Linear Solver Error: {a_err}, relative error: {r_err}")
        return torch.from_numpy(x).to(A.device)
    


# %%
class LM(pp.optim.LevenbergMarquardt):
    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))
            J = modjacrev_vmap(self.model, input)

            # params = dict(self.model.named_parameters())
            # params_values = tuple(params.values())
            # J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
            # for i in range(len(R)):
            #     R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
            #         else self.corrector[i](R = R[i], J = J[i])
            R = R[0]

            # dump J and R
            dump_content = {
                'J': J,
                'R': R,
            }
            J = torch.cat([j.to_sparse_coo() for j in J], dim=-1)
            J_T = J.T @ weight if weight is not None else J.T
            A = (J_T @ J).to_sparse_csr()
            diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(A, op=partial(torch.add, other=(torch.diagonal(A)) * pg['damping']))
            b = -J_T @ R.view(-1, 1)
            dump_content['A'] = A
            dump_content['b'] = b

            torch.save(dump_content, os.path.join(DUMP_DIR, f'{TARGET_DATASET}_{TARGET_PROBLEM}.pt'))
            exit(0)


            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            J_T = J.T @ weight if weight is not None else J.T
            A, self.reject_count = J_T @ J, 0
            # A = A.to_sparse_bsr(blocksize=(1,1))
            A = A.to_sparse_csr()
            diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            while self.last <= self.loss:
                diagonal_op_(A, op=partial(torch.add, other=(torch.diagonal(A)) * pg['damping']))
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
                        p[..., :7] = pp.SE3(p[..., [4,5,6,0,1,2,3]]).add_(pp.se3(d.view(p.shape[0], -1)[..., [3,4,5,0,1,2]]))[..., [3,4,5,6,0,1,2]]
                        if OPTIMIZE_INTRINSICS: p[:, 7:] += d.view(p.shape[0], -1)[:, 6:]
                        continue
                p.add_(d.view(p.shape))

# %%

camera_params_other = None if NUM_CAMERA_PARAMS == trimmed_dataset['camera_params'].shape[1] else trimmed_dataset['camera_params'][:, NUM_CAMERA_PARAMS:]
input = [trimmed_dataset['points_2d'],
         camera_params_other,
         trimmed_dataset['camera_index_of_observations'],
         trimmed_dataset['point_index_of_observations']]

# inverse quat
def openGL2gtsam(pose):
    R = pose.rotation()
    t = pose.translation()
    R90 = torch.eye(3)
    R90[0, 0] = 1
    R90[1, 1] = -1
    R90[2, 2] = -1
    wRc = R.Inv() @ pp.mat2SO3(R90)
    t = R.Inv() @ -t
    # // Our camera-to-world translation wTc = -R'*t
    return pp.SE3(torch.cat([t, wRc], dim=-1))

# gtsam coord
trimmed_dataset['camera_params'][:, :7] = Compose([pp.SE3, openGL2gtsam])(trimmed_dataset['camera_params'][:, [4,5,6,0,1,2,3]])[..., [3,4,5,6,0,1,2]]
trimmed_dataset['points_2d'][:, 1] = -trimmed_dataset['points_2d'][:, 1]

model_non_batched = ReprojNonBatched(trimmed_dataset['camera_params'][:, :NUM_CAMERA_PARAMS].clone(),
                                     trimmed_dataset['points_3d'].clone())

model_non_batched = model_non_batched.to(DEVICE)

# strategy_sparse = TrustRegion(radius=1e5)
strategy_sparse = pp.optim.strategy.Adaptive(damping=0.0001, min=1.5e-9)
#sparse_solver = PCG(tol=1e-3, maxiter=10000)
sparse_solver = SciPySpSolver()
optimizer_sparse = LM(model_non_batched, strategy=strategy_sparse, solver=sparse_solver, reject=30)

# least_square_error(camera_params, points_3d, camera_indices, point_indices, points_2d, intr=None)

print('Starting loss:', least_square_error(model_non_batched.pose, model_non_batched.points_3d, trimmed_dataset['camera_index_of_observations'], trimmed_dataset['point_index_of_observations'], trimmed_dataset['points_2d'], intr=camera_params_other).item())
for idx in range(30):
    loss = optimizer_sparse.step(input)
    print('Loss:', least_square_error(model_non_batched.pose, model_non_batched.points_3d, trimmed_dataset['camera_index_of_observations'], trimmed_dataset['point_index_of_observations'], trimmed_dataset['points_2d'], intr=camera_params_other).item())
print('Ending loss:', least_square_error(model_non_batched.pose, model_non_batched.points_3d, trimmed_dataset['camera_index_of_observations'], trimmed_dataset['point_index_of_observations'], trimmed_dataset['points_2d'], intr=camera_params_other).item())

# %%



