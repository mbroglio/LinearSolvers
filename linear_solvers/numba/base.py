"""Base classes shared by all iterative solvers in ``linear_solvers.numba``.

This module mirrors :mod:`linear_solvers.baseline.base` and provides the same
abstract base class :class:`IterativeSolver` and result dataclass
:class:`SolverResult`.  The Numba subpackage has its own copy so that the two
subpackages remain independent and can be imported without side-effects on each
other.

This module defines:

* :class:`SolverResult` – a lightweight dataclass that packages the full
  outcome of a single :meth:`IterativeSolver.solve` call.
* :class:`IterativeSolver` – the abstract base class that every concrete
  JIT-compiled solver must subclass. It provides common validation helpers
  and enforces a uniform public interface identical to the baseline variant.
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
            solve loop (measured with :func:`time.perf_counter`). Note that
            the Numba first-call JIT compilation cost is **included** in this
            measurement.
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
    """Abstract base class for the Numba-accelerated iterative solvers.

    Every concrete solver in ``linear_solvers.numba`` inherits from this class
    and must implement:

    * The :attr:`name` property (returns a human-readable method label).
    * The :meth:`solve` method (dispatches to a ``@njit`` kernel and wraps
      the result in a :class:`SolverResult`).

    The class also provides two *protected* helper methods used internally by
    the concrete subclasses:

    * :meth:`_validate_inputs` – dimension and shape checks (Python-level).
    * :meth:`_relative_residual` – computes ``‖Ax − b‖₂ / ‖b‖₂``
      (Python-level, used only for the post-loop final check).

    Args:
        tol (float): Convergence tolerance. The solver stops when the relative
            residual ``‖Ax − b‖₂ / ‖b‖₂`` falls below this value.
        max_iter (int, optional): Maximum number of iterations allowed.
            Must be **at least 20 000** (project specification). Defaults to
            ``20000``.

    Raises:
        ValueError: If *max_iter* is less than 20 000.
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

        Used only for the post-loop final residual computation (inside
        JIT-compiled kernels the residual is tracked analytically).

        The relative residual is defined as::

            ‖Ax − b‖₂ / ‖b‖₂

        When ``‖b‖₂ = 0`` the function falls back to the **absolute**
        residual ``‖Ax − b‖₂`` to avoid division by zero.

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
