from time import perf_counter
import numpy as np
from .base import IterativeSolver, SolverResult


class JacobiSolver(IterativeSolver):

    @property
    def name(self):
        return "Jacobi"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = np.diag(A)
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("Jacobi requires non-zero diagonal")

        n = b.shape[0]
        x = np.zeros(n)

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        # Precompute the off-diagonal matrix to avoid matrix allocation natively in the loop
        A_offdiag = A - np.diag(diag)

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
        diag = np.diag(A)
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("Gauss-Seidel requires non-zero diagonal")

        n = A.shape[0]
        x = np.zeros(n)

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        # Optimization: Pre-allocate x_old to avoid creating a new array every iteration
        x_old = np.empty(n)

        for k in range(self.max_iter):
            Ax_b = A @ x - b
            rel_res = np.linalg.norm(Ax_b) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            # Optimization: Copy data into the existing buffer instead of allocating a fresh array
            x_old[:] = x

            for i in range(n):
                x[i] = (
                    b[i]
                    - np.dot(A[i, :i], x[:i])
                    - np.dot(A[i, i + 1 :], x_old[i + 1 :])
                ) / diag[i]

        rel_res = np.linalg.norm(A @ x - b) / norm_b
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class GradientSolver(IterativeSolver):

    @property
    def name(self):
        return "Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if not np.allclose(A, A.T):
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
        if not np.allclose(A, A.T):
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
