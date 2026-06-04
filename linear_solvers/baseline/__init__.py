"""linear_solvers.baseline: Pure NumPy/SciPy iterative solvers.

This subpackage exposes the baseline (non-JIT) implementations of four
classical iterative methods. All solvers operate on sparse matrices stored
in **CSR format** (``scipy.sparse.csr_matrix``).

Public API:
    IterativeSolver: Abstract base class for all iterative solvers.
    SolverResult:    Dataclass carrying the outcome of a solve call.
    JacobiSolver:           Jacobi stationary iteration.
    GaussSeidelSolver:      Gauss-Seidel stationary iteration.
    GradientSolver:         Steepest-descent (gradient) method.
    ConjugateGradientSolver: Conjugate Gradient method (requires SPD matrix).
"""

from .base import IterativeSolver, SolverResult
from .iterative_methods import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)
