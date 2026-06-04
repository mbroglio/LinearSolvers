"""linear_solvers.numba: Numba-JIT-compiled iterative solvers.

This subpackage re-implements the same four iterative methods provided by
:mod:`linear_solvers.baseline`, but replaces the NumPy-level loops with
explicit scalar loops compiled to native machine code via Numba's
``@njit(fastmath=True)`` decorator.

The JIT-compiled kernels operate directly on the raw CSR arrays
(``data``, ``indices``, ``indptr``) passed as plain NumPy arrays, which
allows Numba to generate tight, cache-friendly inner loops without the
Python/NumPy object overhead.

**First-call warm-up**: Numba compiles each kernel the first time it is
invoked with a new type signature. Subsequent calls use the cached
native code and are significantly faster.

Public API:
    IterativeSolver: Abstract base class (re-exported from :mod:`.base`).
    SolverResult:    Result dataclass (re-exported from :mod:`.base`).
    JacobiSolver:           Jacobi stationary iteration (JIT).
    GaussSeidelSolver:      Gauss-Seidel stationary iteration (JIT).
    GradientSolver:         Steepest-descent method (JIT).
    ConjugateGradientSolver: Conjugate Gradient method (JIT).
"""

from .base import IterativeSolver, SolverResult
from .iterative_methods import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)
