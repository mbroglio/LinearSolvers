"""linear_solvers: A Python library for sparse iterative linear solvers.

This package provides two implementations of classical iterative methods for
solving symmetric positive-definite (SPD) linear systems of the form Ax = b,
where A is a sparse matrix in CSR format:

Subpackages:
    baseline: Pure NumPy/SciPy implementations, optimised for readability and
              correctness. Suitable for moderate-size systems and as a reference
              implementation.
    numba:    Numba-JIT-compiled implementations of the same algorithms.
              Significantly faster on large sparse systems thanks to low-level
              loop compilation with ``@njit(fastmath=True)``.

Supported solvers (both subpackages):
    - Jacobi
    - Gauss-Seidel
    - Gradient (steepest-descent)
    - Conjugate Gradient

Typical usage::

    from linear_solvers.baseline import ConjugateGradientSolver
    from scipy.io import mmread

    A = mmread("matrix.mtx").tocsr()
    b = A @ np.ones(A.shape[0])
    solver = ConjugateGradientSolver(tol=1e-8, max_iter=20000)
    result = solver.solve(A, b)
    print(result.converged, result.iterations, result.elapsed_seconds)

Authors:
    Broglio Matteo, Caputo Lorenzo, Giuggioli Daniel
"""
