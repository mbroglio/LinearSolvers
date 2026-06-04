"""Concrete iterative solvers — Numba-JIT-compiled implementation.

This module provides the same four iterative methods as
:mod:`linear_solvers.baseline.iterative_methods`, but offloads the
computationally intensive inner loops to Numba ``@njit(fastmath=True)``
kernels that compile to native machine code on the first invocation.

Public classes:
    JacobiSolver:            Jacobi stationary iteration (JIT-compiled).
    GaussSeidelSolver:       Gauss-Seidel stationary iteration (JIT-compiled).
    GradientSolver:          Steepest-descent method (JIT-compiled).
    ConjugateGradientSolver: Conjugate Gradient method (JIT-compiled).

Private JIT kernels (module-level, not part of the public API):
    _csr_matvec:  Sparse CSR matrix-vector product.
    _jacobi_core: Full Jacobi iteration loop.
    _gs_core:     Full Gauss-Seidel iteration loop.
    _grad_core:   Full steepest-descent iteration loop.
    _cg_core:     Full Conjugate Gradient iteration loop.

Design rationale:
    Each public solver class handles Python-level validation (symmetry checks,
    diagonal checks, norm-b guard) and timing, then delegates all arithmetic
    to the corresponding ``@njit`` function.  This separation keeps the JIT
    kernels free of Python objects, which is required for Numba's nopython
    mode (``nopython=True`` implied by ``@njit``).

Note on ``fastmath=True``:
    This flag permits the compiler to reorder floating-point operations for
    speed (e.g., fused multiply-add), which may introduce small changes
    relative to IEEE-754 strict mode.  Results are accurate for well-
    conditioned problems but may differ slightly from the baseline for
    ill-conditioned or nearly-singular matrices.
"""

from time import perf_counter

import numpy as np
from numba import njit

from .base import IterativeSolver, SolverResult


# ---------------------------------------------------------------------------
# Shared JIT kernel: CSR matrix-vector product
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _csr_matvec(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n: int,
    x: np.ndarray,
) -> np.ndarray:
    """Compute the sparse CSR matrix-vector product *y = A x*.

    This kernel is shared by :func:`_grad_core` and :func:`_cg_core` to avoid
    code duplication.  It iterates over each row of the CSR representation and
    accumulates the dot product explicitly, enabling Numba to produce tight,
    SIMD-friendly machine code.

    Args:
        data (numpy.ndarray): Non-zero values of *A*, shape ``(nnz,)``.
        indices (numpy.ndarray): Column indices of each non-zero, shape ``(nnz,)``.
        indptr (numpy.ndarray): Row pointer array, shape ``(n+1,)``. Row *i*
            spans ``data[indptr[i]:indptr[i+1]]``.
        n (int): Number of rows (and columns) of *A*.
        x (numpy.ndarray): Input vector, shape ``(n,)``.

    Returns:
        numpy.ndarray: Output vector *y = Ax* of shape ``(n,)``.
    """
    res = np.zeros(n)
    for i in range(n):
        dot = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            dot += data[p] * x[indices[p]]
        res[i] = dot
    return res


# ---------------------------------------------------------------------------
# Jacobi JIT kernel
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _jacobi_core(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    b: np.ndarray,
    diag: np.ndarray,
    n: int,
    max_iter: int,
    tol: float,
    norm_b: float,
):
    """JIT-compiled Jacobi iteration kernel.

    Implements the full Jacobi loop operating directly on CSR raw arrays.
    Each iteration computes a full ``Ax`` product row-by-row, evaluates the
    residual norm, checks for convergence, and then updates all components
    simultaneously using a temporary buffer ``x_new``.

    Simultaneous update (unlike Gauss-Seidel): the new iterate ``x_new[i]``
    is computed from the **old** ``x``, so all row computations within one
    sweep are independent of each other.

    Args:
        data (numpy.ndarray): Non-zero values of *A*.
        indices (numpy.ndarray): Column indices of the non-zeros.
        indptr (numpy.ndarray): CSR row pointer array.
        b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.
        diag (numpy.ndarray): Diagonal entries of *A*, shape ``(n,)``.
        n (int): System size.
        max_iter (int): Maximum number of iterations.
        tol (float): Relative residual tolerance.
        norm_b (float): Pre-computed ``‖b‖₂`` (passed from the Python caller
            to avoid recomputation inside the JIT context).

    Returns:
        tuple: A 4-tuple ``(x, k, rel_res, converged)`` where:

        * ``x`` (numpy.ndarray) – approximate solution at termination.
        * ``k`` (int) – iteration index at which the solver stopped.
        * ``rel_res`` (float) – relative residual at termination.
        * ``converged`` (bool) – ``True`` if tolerance was reached.
    """
    x = np.zeros(n)      # initial iterate x^0 = 0
    x_new = np.zeros(n)  # temporary buffer for the simultaneous update

    for k in range(max_iter):
        # --- convergence check: compute ‖Ax − b‖₂ row by row ---
        norm_res = 0.0
        for i in range(n):
            Ax_i = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                Ax_i += data[p] * x[indices[p]]
            res_i = Ax_i - b[i]
            norm_res += res_i * res_i
            # Jacobi update for component i: x_new[i] = x[i] - (Ax_i - b[i]) / d_ii
            x_new[i] = x[i] - res_i / diag[i]

        rel_res = np.sqrt(norm_res) / norm_b  # ‖Ax − b‖₂ / ‖b‖₂
        if rel_res < tol:
            return x_new, k, rel_res, True

        # Copy x_new into x for the next iteration (simultaneous update).
        for i in range(n):
            x[i] = x_new[i]

    # --- final residual after max_iter iterations ---
    norm_res = 0.0
    for i in range(n):
        Ax_i = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            Ax_i += data[p] * x[indices[p]]
        res_i = Ax_i - b[i]
        norm_res += res_i * res_i
    rel_res = np.sqrt(norm_res) / norm_b
    return x, max_iter, rel_res, False


# ---------------------------------------------------------------------------
# Jacobi solver class
# ---------------------------------------------------------------------------

class JacobiSolver(IterativeSolver):
    """Jacobi stationary iteration — Numba-accelerated.

    Thin Python wrapper around :func:`_jacobi_core`. Performs Python-level
    validation, extracts the CSR raw arrays, and delegates the iteration loop
    to the JIT-compiled kernel.

    See :class:`linear_solvers.baseline.JacobiSolver` for a full algorithmic
    description.

    Args:
        tol (float): Convergence tolerance on the relative residual.
        max_iter (int, optional): Maximum number of iterations. Must be
            ``>= 20000``. Defaults to ``20000``.
    """

    @property
    def name(self) -> str:
        """Return the solver label.

        Returns:
            str: ``"Jacobi"``.
        """
        return "Jacobi"

    def solve(self, A, b: np.ndarray) -> SolverResult:
        """Run the JIT-compiled Jacobi iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square SPD (or strictly diagonally
                dominant) coefficient matrix of shape ``(n, n)`` in CSR
                format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields ``solution``, ``iterations``,
            ``elapsed_seconds``, ``relative_residual``, ``converged``.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, or any
                diagonal entry satisfies ``|d_ii| < 1e-15``.

        Note:
            The first call triggers Numba JIT compilation; the measured
            ``elapsed_seconds`` includes this one-time compilation cost.
        """
        self._validate_inputs(A, b)

        diag = A.diagonal()
        if np.any(np.abs(diag) < 1e-15):
            raise ValueError("Jacobi requires non-zero diagonal")

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()
        x, k, rel_res, conv = _jacobi_core(
            A.data,
            A.indices,
            A.indptr,
            b,
            diag,
            b.shape[0],
            self.max_iter,
            self.tol,
            norm_b,
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


# ---------------------------------------------------------------------------
# Gauss-Seidel JIT kernel
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _gs_core(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    b: np.ndarray,
    diag: np.ndarray,
    n: int,
    max_iter: int,
    tol: float,
    norm_b: float,
):
    """JIT-compiled Gauss-Seidel iteration kernel.

    Unlike :func:`_jacobi_core`, the component update uses the **latest**
    values of ``x``, so each updated ``x[i]`` is immediately visible to
    subsequent components within the same sweep.

    The convergence check is performed at the beginning of each sweep before
    any components are updated, so the residual reflects the state of *x* at
    the end of the previous sweep.

    Args:
        data (numpy.ndarray): Non-zero values of *A*.
        indices (numpy.ndarray): Column indices of the non-zeros.
        indptr (numpy.ndarray): CSR row pointer array.
        b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.
        diag (numpy.ndarray): Diagonal entries of *A*, shape ``(n,)``.
        n (int): System size.
        max_iter (int): Maximum number of iterations.
        tol (float): Relative residual tolerance.
        norm_b (float): Pre-computed ``‖b‖₂``.

    Returns:
        tuple: ``(x, k, rel_res, converged)`` — same convention as
        :func:`_jacobi_core`.
    """
    x = np.zeros(n)  # initial iterate x^0 = 0

    for k in range(max_iter):
        # --- convergence check at the start of each sweep ---
        norm_res = 0.0
        for i in range(n):
            Ax_i = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                Ax_i += data[p] * x[indices[p]]
            res_i = Ax_i - b[i]
            norm_res += res_i * res_i

        rel_res = np.sqrt(norm_res) / norm_b  # ‖Ax − b‖₂ / ‖b‖₂
        if rel_res < tol:
            return x, k, rel_res, True

        # --- Gauss-Seidel sweep: update x[i] using the latest x values ---
        for i in range(n):
            sum_Ax = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if i != j:  # exclude the diagonal term
                    sum_Ax += data[p] * x[j]
            x[i] = (b[i] - sum_Ax) / diag[i]

    # --- final residual after max_iter iterations ---
    norm_res = 0.0
    for i in range(n):
        Ax_i = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            Ax_i += data[p] * x[indices[p]]
        res_i = Ax_i - b[i]
        norm_res += res_i * res_i
    rel_res = np.sqrt(norm_res) / norm_b
    return x, max_iter, rel_res, False


# ---------------------------------------------------------------------------
# Gauss-Seidel solver class
# ---------------------------------------------------------------------------

class GaussSeidelSolver(IterativeSolver):
    """Gauss-Seidel stationary iteration — Numba-accelerated.

    Thin Python wrapper around :func:`_gs_core`. Performs Python-level
    validation, extracts the CSR raw arrays, and delegates the iteration loop
    to the JIT-compiled kernel.

    See :class:`linear_solvers.baseline.GaussSeidelSolver` for a full
    algorithmic description.

    Args:
        tol (float): Convergence tolerance on the relative residual.
        max_iter (int, optional): Maximum number of iterations. Must be
            ``>= 20000``. Defaults to ``20000``.
    """

    @property
    def name(self) -> str:
        """Return the solver label.

        Returns:
            str: ``"Gauss-Seidel"``.
        """
        return "Gauss-Seidel"

    def solve(self, A, b: np.ndarray) -> SolverResult:
        """Run the JIT-compiled Gauss-Seidel iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square SPD (or strictly diagonally
                dominant) coefficient matrix of shape ``(n, n)`` in CSR
                format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields ``solution``, ``iterations``,
            ``elapsed_seconds``, ``relative_residual``, ``converged``.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, or any
                diagonal entry satisfies ``|d_ii| < 1e-15``.

        Note:
            The first call triggers Numba JIT compilation; the measured
            ``elapsed_seconds`` includes this one-time compilation cost.
        """
        self._validate_inputs(A, b)

        diag = A.diagonal()
        if np.any(np.abs(diag) < 1e-15):
            raise ValueError("Gauss-Seidel requires non-zero diagonal")

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()
        x, k, rel_res, conv = _gs_core(
            A.data,
            A.indices,
            A.indptr,
            b,
            diag,
            A.shape[0],
            self.max_iter,
            self.tol,
            norm_b,
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


# ---------------------------------------------------------------------------
# Gradient (steepest descent) JIT kernel
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _grad_core(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    b: np.ndarray,
    n: int,
    max_iter: int,
    tol: float,
    norm_b: float,
):
    """JIT-compiled steepest-descent iteration kernel.

    Maintains the residual *r* analytically (``r ← r − α·Ar``) so that each
    iteration requires exactly one call to :func:`_csr_matvec` rather than
    re-computing ``Ax`` from scratch.

    Positive definiteness is verified implicitly: if ``r^T A r ≤ 0`` is
    encountered, the solver exits early with ``converged=False`` rather than
    raising an exception (Numba ``@njit`` functions cannot raise Python
    exceptions).  The Python-level caller handles strict SPD validation.

    Args:
        data (numpy.ndarray): Non-zero values of *A*.
        indices (numpy.ndarray): Column indices of the non-zeros.
        indptr (numpy.ndarray): CSR row pointer array.
        b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.
        n (int): System size.
        max_iter (int): Maximum number of iterations.
        tol (float): Relative residual tolerance.
        norm_b (float): Pre-computed ``‖b‖₂``.

    Returns:
        tuple: ``(x, k, rel_res, converged)`` — same convention as
        :func:`_jacobi_core`.
    """
    x = np.zeros(n)  # initial iterate x^0 = 0
    r = b.copy()     # initial residual r^0 = b − A·0 = b

    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b  # ‖r_k‖₂ / ‖b‖₂
        if rel_res < tol:
            return x, k, rel_res, True

        Ar = _csr_matvec(data, indices, indptr, n, r)  # single mat-vec per iter
        denom = np.dot(r, Ar)  # r^T A r — must be > 0 for SPD matrices
        if denom <= 0.0:
            return x, k, rel_res, False  # non-SPD matrix detected; exit early

        alpha = np.dot(r, r) / denom  # optimal step length α_k = ‖r‖² / (r^T A r)
        x = x + alpha * r
        r = r - alpha * Ar

    rel_res = np.linalg.norm(r) / norm_b  # final residual after max_iter iterations
    return x, max_iter, rel_res, False


# ---------------------------------------------------------------------------
# Gradient solver class
# ---------------------------------------------------------------------------

class GradientSolver(IterativeSolver):
    """Steepest-descent (gradient) method — Numba-accelerated.

    Thin Python wrapper around :func:`_grad_core`. Performs Python-level
    symmetry validation, extracts the CSR raw arrays, and delegates the
    iteration loop to the JIT-compiled kernel.

    See :class:`linear_solvers.baseline.GradientSolver` for a full algorithmic
    description.

    Args:
        tol (float): Convergence tolerance on the relative residual.
        max_iter (int, optional): Maximum number of iterations. Must be
            ``>= 20000``. Defaults to ``20000``.
    """

    @property
    def name(self) -> str:
        """Return the solver label.

        Returns:
            str: ``"Gradient"``.
        """
        return "Gradient"

    def solve(self, A, b: np.ndarray) -> SolverResult:
        """Run the JIT-compiled steepest-descent iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square, **symmetric** and positive
                definite coefficient matrix of shape ``(n, n)`` in CSR format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields ``solution``, ``iterations``,
            ``elapsed_seconds``, ``relative_residual``, ``converged``.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, or *A* is
                not symmetric (checked with ``max |A − Aᵀ| > 1e-10``).

        Note:
            The first call triggers Numba JIT compilation; the measured
            ``elapsed_seconds`` includes this one-time compilation cost.
        """
        self._validate_inputs(A, b)

        if np.abs(A - A.T).max() > 1e-10:
            raise ValueError("Gradient requires symmetric matrix")

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()
        x, k, rel_res, conv = _grad_core(
            A.data, A.indices, A.indptr, b, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


# ---------------------------------------------------------------------------
# Conjugate Gradient JIT kernel
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _cg_core(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    b: np.ndarray,
    n: int,
    max_iter: int,
    tol: float,
    norm_b: float,
):
    """JIT-compiled Conjugate Gradient iteration kernel.

    Implements the standard CG algorithm with A-conjugate search directions.
    The residual and search direction are propagated analytically, so each
    iteration requires one call to :func:`_csr_matvec` and three dot products.

    In exact arithmetic the method converges in at most *n* steps; in
    floating-point arithmetic superlinear convergence is observed for SPD
    matrices with clustered eigenvalues.

    If ``p^T A p ≤ 0`` is encountered (non-SPD matrix), the kernel exits
    early with ``converged=False`` rather than raising an exception (see note
    in :func:`_grad_core`).

    Args:
        data (numpy.ndarray): Non-zero values of *A*.
        indices (numpy.ndarray): Column indices of the non-zeros.
        indptr (numpy.ndarray): CSR row pointer array.
        b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.
        n (int): System size.
        max_iter (int): Maximum number of iterations.
        tol (float): Relative residual tolerance.
        norm_b (float): Pre-computed ``‖b‖₂``.

    Returns:
        tuple: ``(x, k, rel_res, converged)`` — same convention as
        :func:`_jacobi_core`.
    """
    x = np.zeros(n)  # initial iterate x^0 = 0
    r = b.copy()     # initial residual r^0 = b − A·0 = b
    p = r.copy()     # initial search direction p^0 = r^0

    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b  # ‖r_k‖₂ / ‖b‖₂
        if rel_res < tol:
            return x, k, rel_res, True

        Ap = _csr_matvec(data, indices, indptr, n, p)  # single mat-vec per iter
        denom = np.dot(p, Ap)  # p^T A p — must be > 0 for SPD matrices
        if denom <= 0.0:
            return x, k, rel_res, False  # non-SPD matrix detected; exit early

        rr_old = np.dot(r, r)     # ‖r_k‖₂² before the update
        alpha = rr_old / denom    # step length α_k = ‖r_k‖² / (p_k^T A p_k)

        x = x + alpha * p
        r = r - alpha * Ap

        rr_new = np.dot(r, r)         # ‖r_{k+1}‖₂² after the update
        beta = rr_new / rr_old        # Fletcher–Reeves coefficient β_k
        p = r + beta * p              # new A-conjugate search direction

    rel_res = np.linalg.norm(r) / norm_b  # final residual after max_iter iterations
    return x, max_iter, rel_res, False


# ---------------------------------------------------------------------------
# Conjugate Gradient solver class
# ---------------------------------------------------------------------------

class ConjugateGradientSolver(IterativeSolver):
    """Conjugate Gradient method — Numba-accelerated.

    Thin Python wrapper around :func:`_cg_core`. Performs Python-level
    symmetry validation, extracts the CSR raw arrays, and delegates the
    iteration loop to the JIT-compiled kernel.

    See :class:`linear_solvers.baseline.ConjugateGradientSolver` for a full
    algorithmic description.

    Args:
        tol (float): Convergence tolerance on the relative residual.
        max_iter (int, optional): Maximum number of iterations. Must be
            ``>= 20000``. Defaults to ``20000``.
    """

    @property
    def name(self) -> str:
        """Return the solver label.

        Returns:
            str: ``"Conjugate Gradient"``.
        """
        return "Conjugate Gradient"

    def solve(self, A, b: np.ndarray) -> SolverResult:
        """Run the JIT-compiled Conjugate Gradient iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square, **symmetric positive
                definite** coefficient matrix of shape ``(n, n)`` in CSR
                format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields ``solution``, ``iterations``,
            ``elapsed_seconds``, ``relative_residual``, ``converged``.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, or *A* is
                not symmetric (checked with ``max |A − Aᵀ| > 1e-10``).

        Note:
            The first call triggers Numba JIT compilation; the measured
            ``elapsed_seconds`` includes this one-time compilation cost.
        """
        self._validate_inputs(A, b)

        if np.abs(A - A.T).max() > 1e-10:
            raise ValueError("Conjugate Gradient requires symmetric matrix")

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()
        x, k, rel_res, conv = _cg_core(
            A.data, A.indices, A.indptr, b, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)
