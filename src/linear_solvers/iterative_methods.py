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
        start = perf_counter()
        
        for k in range(self.max_iter):
            rel_res = self._relative_residual(A, x, b)
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)
            x = (b - (A - np.diag(diag)) @ x) / diag

        rel_res = self._relative_residual(A, x, b)
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
        start = perf_counter()
        
        for k in range(self.max_iter):
            rel_res = self._relative_residual(A, x, b)
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)
            
            x_old = x.copy()
            for i in range(n):
                x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / diag[i]

        rel_res = self._relative_residual(A, x, b)
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
        start = perf_counter()
        
        for k in range(self.max_iter):
            rel_res = self._relative_residual(A, x, b)
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)
            
            Ar = A @ r
            denom = np.dot(r, Ar)
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")
            
            alpha = np.dot(r, r) / denom
            x = x + alpha * r
            r = r - alpha * Ar

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
        start = perf_counter()
        
        for k in range(self.max_iter):
            rel_res = self._relative_residual(A, x, b)
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)
            
            Ap = A @ p
            denom = np.dot(p, Ap)
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")
            
            rr_old = np.dot(r, r)
            alpha = rr_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = np.dot(r, r)
            beta = rr_new / rr_old
            p = r + beta * p

        rel_res = self._relative_residual(A, x, b)
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)
