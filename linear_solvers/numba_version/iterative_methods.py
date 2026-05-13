from time import perf_counter
import numpy as np
from numba import njit
from .base import IterativeSolver, SolverResult


@njit(fastmath=True)
def _jacobi_core(A, b, diag, n, max_iter, tol, norm_b):
    x = np.zeros(n)
    x_new = np.zeros(n)
    for k in range(max_iter):
        norm_res = 0.0
        for i in range(n):
            Ax_i = 0.0
            for j in range(n):
                Ax_i += A[i, j] * x[j]
            res_i = Ax_i - b[i]
            norm_res += res_i * res_i
            x_new[i] = x[i] - res_i / diag[i]

        rel_res = np.sqrt(norm_res) / norm_b
        if rel_res < tol:
            return x_new, k, rel_res, True

        for i in range(n):
            x[i] = x_new[i]

    norm_res = 0.0
    for i in range(n):
        Ax_i = 0.0
        for j in range(n):
            Ax_i += A[i, j] * x[j]
        res_i = Ax_i - b[i]
        norm_res += res_i * res_i
    rel_res = np.sqrt(norm_res) / norm_b
    return x, max_iter, rel_res, False


class JacobiSolver(IterativeSolver):
    @property
    def name(self):
        return "Jacobi"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = np.diag(A)
        if np.any(np.abs(diag) < 1e-15):
            raise ValueError("Jacobi requires non-zero diagonal")
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _jacobi_core(
            A, b, diag, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


@njit(fastmath=True)
def _gs_core(A, b, diag, n, max_iter, tol, norm_b):
    x = np.zeros(n)
    for k in range(max_iter):
        norm_res = 0.0
        for i in range(n):
            Ax_i = 0.0
            for j in range(n):
                Ax_i += A[i, j] * x[j]
            res_i = Ax_i - b[i]
            norm_res += res_i * res_i

        rel_res = np.sqrt(norm_res) / norm_b
        if rel_res < tol:
            return x, k, rel_res, True

        for i in range(n):
            sum_Ax = 0.0
            for j in range(n):
                if i != j:
                    sum_Ax += A[i, j] * x[j]
            x[i] = (b[i] - sum_Ax) / diag[i]

    norm_res = 0.0
    for i in range(n):
        Ax_i = 0.0
        for j in range(n):
            Ax_i += A[i, j] * x[j]
        res_i = Ax_i - b[i]
        norm_res += res_i * res_i
    rel_res = np.sqrt(norm_res) / norm_b
    return x, max_iter, rel_res, False


class GaussSeidelSolver(IterativeSolver):
    @property
    def name(self):
        return "Gauss-Seidel"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = np.diag(A)
        if np.any(np.abs(diag) < 1e-15):
            raise ValueError("Gauss-Seidel requires non-zero diagonal")
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _gs_core(
            A, b, diag, A.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


@njit(fastmath=True)
def _grad_core(A, b, n, max_iter, tol, norm_b):
    x = np.zeros(n)
    r = b.copy()
    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b
        if rel_res < tol:
            return x, k, rel_res, True
        Ar = A @ r
        denom = np.dot(r, Ar)
        if denom <= 0.0:
            return x, k, rel_res, False
        alpha = np.dot(r, r) / denom
        x = x + alpha * r
        r = r - alpha * Ar
    rel_res = np.linalg.norm(r) / norm_b
    return x, max_iter, rel_res, False


class GradientSolver(IterativeSolver):
    @property
    def name(self):
        return "Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if not np.allclose(A, A.T):
            raise ValueError("Gradient requires symmetric matrix")
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _grad_core(
            A, b, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


@njit(fastmath=True)
def _cg_core(A, b, n, max_iter, tol, norm_b):
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b
        if rel_res < tol:
            return x, k, rel_res, True
        Ap = A @ p
        denom = np.dot(p, Ap)
        if denom <= 0.0:
            return x, k, rel_res, False
        rr_old = np.dot(r, r)
        alpha = rr_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = np.dot(r, r)
        beta = rr_new / rr_old
        p = r + beta * p
    rel_res = np.linalg.norm(r) / norm_b
    return x, max_iter, rel_res, False


class ConjugateGradientSolver(IterativeSolver):
    @property
    def name(self):
        return "Conjugate Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if not np.allclose(A, A.T):
            raise ValueError("Conjugate Gradient requires symmetric matrix")
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _cg_core(
            A, b, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)
