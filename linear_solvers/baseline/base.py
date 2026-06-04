"""Base classes shared by all iterative solvers in ``linear_solvers.baseline``.

This module defines:

* :class:`SolverResult` – a lightweight dataclass that packages the full
  outcome of a single :meth:`IterativeSolver.solve` call.
* :class:`IterativeSolver` – the abstract base class that every concrete
  solver must subclass. It provides common validation helpers and enforces
  a uniform public interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# SolverResult
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Immutable record of a completed solver run.

    Attributes:
        solution (numpy.ndarray): The approximate solution vector *x* of shape
            ``(n,)``, where *n* is the number of unknowns.
        iterations (int): Number of iterations executed before the solver
            stopped (either convergence or ``max_iter`` reached).
        elapsed_seconds (float): Wall-clock time, in seconds, spent inside the
            solve loop (measured with :func:`time.perf_counter`).
        relative_residual (float): Relative residual at termination,
            defined as ``‖Ax − b‖₂ / ‖b‖₂`` (or ``‖Ax − b‖₂`` when
            ``‖b‖₂ = 0``).
        converged (bool): ``True`` if the relative residual dropped below the
            requested tolerance before ``max_iter`` iterations were exhausted.
    """

    solution: np.ndarray
    iterations: int
    elapsed_seconds: float
    relative_residual: float
    converged: bool


# ---------------------------------------------------------------------------
# IterativeSolver
# ---------------------------------------------------------------------------

class IterativeSolver(ABC):
    """Abstract base class for stationary and Krylov iterative solvers.

    Every concrete solver in ``linear_solvers.baseline`` inherits from this
    class and must implement:

    * The :attr:`name` property (returns a human-readable method label).
    * The :meth:`solve` method (runs the iteration and returns a
      :class:`SolverResult`).

    The class also provides two *protected* helper methods used internally by
    the concrete subclasses:

    * :meth:`_validate_inputs` – dimension and shape checks.
    * :meth:`_relative_residual` – computes ``‖Ax − b‖₂ / ‖b‖₂``.

    Args:
        tol (float): Convergence tolerance. The solver stops when the relative
            residual ``‖Ax − b‖₂ / ‖b‖₂`` falls below this value.
        max_iter (int, optional): Maximum number of iterations allowed.
            Must be **at least 20 000** (project specification). Defaults to
            ``20000``.

    Raises:
        ValueError: If *max_iter* is less than 20 000.

    Example::

        class MySolver(IterativeSolver):
            @property
            def name(self):
                return "My Solver"

            def solve(self, A, b):
                self._validate_inputs(A, b)
                ...
    """

    def __init__(self, tol: float, max_iter: int = 20000) -> None:
        if max_iter < 20000:
            raise ValueError("max_iter must be >= 20000")
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the iterative method.

        Returns:
            str: A short descriptive string (e.g. ``"Jacobi"``).
        """
        pass

    @abstractmethod
    def solve(self, A, b: np.ndarray) -> SolverResult:
        """Solve the linear system *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square sparse coefficient matrix of
                shape ``(n, n)`` in CSR format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Object containing the approximate solution, iteration
            count, elapsed time, relative residual, and convergence flag.
        """
        pass

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, A, b: np.ndarray) -> None:
        """Check that *A* and *b* have compatible shapes.

        Args:
            A: Coefficient matrix. Must be two-dimensional and square.
            b (numpy.ndarray): Right-hand-side vector. Must be one-dimensional
                with length equal to the number of rows of *A*.

        Raises:
            ValueError: If *A* is not a square 2-D array, or if the dimensions
                of *A* and *b* do not match.
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if b.ndim != 1 or A.shape[0] != b.shape[0]:
            raise ValueError("Dimension mismatch")

    def _relative_residual(self, A, x: np.ndarray, b: np.ndarray) -> float:
        """Compute the relative (or absolute) residual of the current iterate.

        The relative residual is defined as::

            ‖Ax − b‖₂ / ‖b‖₂

        When ``‖b‖₂ = 0`` (i.e. the right-hand side is the zero vector) the
        function falls back to the **absolute** residual ``‖Ax − b‖₂`` to
        avoid division by zero.

        Args:
            A: Coefficient matrix.
            x (numpy.ndarray): Current iterate (approximate solution).
            b (numpy.ndarray): Right-hand-side vector.

        Returns:
            float: Relative residual ``‖Ax − b‖₂ / ‖b‖₂``, or absolute
            residual ``‖Ax − b‖₂`` if ``‖b‖₂ = 0``.
        """
        norm_b = np.linalg.norm(b)
        if norm_b > 0:
            return np.linalg.norm(A @ x - b) / norm_b
        return np.linalg.norm(A @ x - b)
