from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverResult:
    solution: np.ndarray
    iterations: int
    elapsed_seconds: float
    relative_residual: float
    converged: bool


class IterativeSolver(ABC):

    def __init__(self, tol, max_iter=20000):
        if max_iter < 20000:
            raise ValueError("max_iter must be >= 20000")
        self.tol = tol
        self.max_iter = max_iter

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def solve(self, A, b):
        pass

    def _validate_inputs(self, A, b):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if b.ndim != 1 or A.shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch")

    def _relative_residual(self, A, x, b):
        norm_b = np.linalg.norm(b)
        if norm_b > 0:
            return np.linalg.norm(A @ x - b) / norm_b
        return np.linalg.norm(A @ x - b)
