from time import perf_counter
import numpy as np
from .base import IterativeSolver, SolverResult


class JacobiSolver(IterativeSolver):

    @property
    def name(self):
        return "Jacobi"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = A.diagonal()
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("Jacobi requires non-zero diagonal")

        n = b.shape[0]
        x = np.zeros(n)

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        # Precompute the off-diagonal matrix to avoid matrix allocation natively in the loop
        A_offdiag = A.copy()
        A_offdiag.setdiag(0)

        for k in range(self.max_iter):
            Ax_b = A @ x - b
            rel_res = np.linalg.norm(Ax_b) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            # Optimization: Write result into existing array x instead of creating a new one
            x[:] = (b - A_offdiag @ x) / diag

        rel_res = np.linalg.norm(A @ x - b) / norm_b
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class GaussSeidelSolver(IterativeSolver):

    @property
    def name(self):
        return "Gauss-Seidel"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = A.diagonal()
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("Gauss-Seidel requires non-zero diagonal")

        n = A.shape[0]
        x = np.zeros(n)

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        for k in range(self.max_iter):
            Ax_b = A @ x - b
            rel_res = np.linalg.norm(Ax_b) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            for i in range(n):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                data = A.data[row_start:row_end]
                indices = A.indices[row_start:row_end]

                # Optimization: Vectorized dot product mapped exactly to memory indices
                sum_Ax = np.dot(data, x[indices]) - diag[i] * x[i]
                x[i] = (b[i] - sum_Ax) / diag[i]

        rel_res = np.linalg.norm(A @ x - b) / norm_b
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class GradientSolver(IterativeSolver):

    @property
    def name(self):
        return "Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if np.abs(A - A.T).max() > 1e-10:
            raise ValueError("Gradient requires symmetric matrix")

        n = b.shape[0]
        x = np.zeros(n)
        r = b.copy()

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        for k in range(self.max_iter):
            # Optimization: r is already the residual (b - Ax), no need to recompute A @ x
            rel_res = np.linalg.norm(r) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            Ar = A @ r
            denom = np.dot(r, Ar)
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")

            alpha = np.dot(r, r) / denom

            # Optimization: In-place arithmetic avoids array reallocation
            x += alpha * r
            r -= alpha * Ar

        rel_res = self._relative_residual(A, x, b)
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class ConjugateGradientSolver(IterativeSolver):

    @property
    def name(self):
        return "Conjugate Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if np.abs(A - A.T).max() > 1e-10:
            raise ValueError("Conjugate Gradient requires symmetric matrix")

        n = b.shape[0]
        x = np.zeros(n)
        r = b.copy()
        p = r.copy()

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        for k in range(self.max_iter):
            # Optimization: r is already the residual (b - Ax), no need to recompute A @ x
            rel_res = np.linalg.norm(r) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            Ap = A @ p
            denom = np.dot(p, Ap)
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")

            rr_old = np.dot(r, r)
            alpha = rr_old / denom

            # Optimization: In-place arithmetic avoids array reallocation
            x += alpha * p
            r -= alpha * Ap

            rr_new = np.dot(r, r)
            beta = rr_new / rr_old

            # Optimization: Update p in-place
            p *= beta
            p += r

        rel_res = self._relative_residual(A, x, b)
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)
