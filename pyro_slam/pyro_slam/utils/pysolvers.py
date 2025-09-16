import torch

from pypose.optim.solver import CG
from pyro_slam.sparse.py_ops import spdiags_

from pyro_slam.sparse.solve import CuDirectSparseSolver as cuSolverSP


class PCG(CG):
    def __init__(self, maxiter=None, tol=1e-5):
        super().__init__(maxiter, tol)
    def forward(self, A, b, x=None, M=None) -> torch.Tensor:
        if b.dim() == 1:
            b = b[..., None]
        l_diag = A.diagonal()
        l_diag[l_diag.abs() < 1e-6] = 1e-6
        M = spdiags_((1 / l_diag), None, shape=A.shape, layout=None)
        if A.layout == torch.sparse_csr:
            # M = M.to_sparse_csr()
            pass
            # A = M @ A
        elif A.layout == torch.sparse_bsr:
            M = M.to_sparse_bsr(blocksize=A.values().shape[-2:]).to(A.device)
            # A = M @ A.to_sparse_bsc(blocksize=A.values().shape[-2:])
        # b = M @ b

        res = super().forward(A, b, x, M)
        res = res.squeeze(-1) 
        return res

class SciPySpSolver(torch.nn.Module):
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