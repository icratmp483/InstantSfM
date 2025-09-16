import numpy as np
from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky

class L1Solver:
    def __init__(self, mat):
        self.a_ = mat
        spd_mat = self.a_.T @ self.a_
        self.linear_solver_ = cholesky(csc_matrix(spd_mat))

    def solve(self, rhs, L1_SOLVER_OPTIONS):
        x = np.zeros(self.a_.shape[1])
        z = np.zeros(self.a_.shape[0])
        u = np.zeros(self.a_.shape[0])

        rhs_norm = np.linalg.norm(rhs)
        primal_abs_tolerance_eps = np.sqrt(self.a_.shape[0]) * L1_SOLVER_OPTIONS['absolute_tolerance']
        dual_abs_tolerance_eps = np.sqrt(self.a_.shape[1]) * L1_SOLVER_OPTIONS['absolute_tolerance']

        for _ in range(L1_SOLVER_OPTIONS['max_num_iterations']):
            x = self.linear_solver_(self.a_.T @ (rhs + z - u))
            
            a_times_x = self.a_ @ x
            ax_hat = L1_SOLVER_OPTIONS['alpha'] * a_times_x + (1.0 - L1_SOLVER_OPTIONS['alpha']) * (z + rhs)

            z_old = z.copy()
            z = self.shrinkage(ax_hat - rhs + u, 1.0 / L1_SOLVER_OPTIONS['rho'])

            u += ax_hat - z - rhs

            r_norm = np.linalg.norm(a_times_x - z - rhs)
            s_norm = np.linalg.norm(-L1_SOLVER_OPTIONS['rho'] * self.a_.T @ (z - z_old))
            max_norm = max(np.linalg.norm(a_times_x), np.linalg.norm(z), rhs_norm)
            primal_eps = primal_abs_tolerance_eps + L1_SOLVER_OPTIONS['relative_tolerance'] * max_norm
            dual_eps = dual_abs_tolerance_eps + L1_SOLVER_OPTIONS['relative_tolerance'] * np.linalg.norm(L1_SOLVER_OPTIONS['rho'] * self.a_.T @ u)

            if r_norm < primal_eps and s_norm < dual_eps:
                break
        return x
    
    @staticmethod
    def shrinkage(vec, kappa):
        return np.maximum(0, vec - kappa) - np.maximum(0, -vec - kappa)