"""Concrete iterative solvers — baseline (pure NumPy/SciPy) implementation.

This module provides four classical iterative methods for solving sparse
symmetric positive-definite (SPD) linear systems ``Ax = b``:

* :class:`JacobiSolver`           – Jacobi stationary iteration.
* :class:`GaussSeidelSolver`      – Gauss-Seidel stationary iteration.
* :class:`GradientSolver`         – Steepest-descent (gradient) method.
* :class:`ConjugateGradientSolver`– Conjugate Gradient (CG) method.

All solvers expect the coefficient matrix *A* to be a
``scipy.sparse.csr_matrix``. The stopping criterion is the **relative
residual**:

    ‖Ax − b‖₂ / ‖b‖₂ < tol

Implementation notes:
    - The Jacobi update avoids building a full off-diagonal matrix at each
      step by pre-computing it once before the loop.
    - Gauss-Seidel accesses CSR row data directly to compute the dot product
      without storing additional sparse structures.
    - Gradient and Conjugate Gradient maintain the residual vector
      analytically (``r ← r − α·Ar``) to avoid recomputing ``Ax`` at each
      step, reducing the per-iteration cost to a single sparse matrix-vector
      product.
"""

from time import perf_counter

import numpy as np

from .base import IterativeSolver, SolverResult


# ---------------------------------------------------------------------------
# Jacobi
# ---------------------------------------------------------------------------

class JacobiSolver(IterativeSolver):
    """Jacobi stationary iteration for sparse linear systems.

    The Jacobi method decomposes *A* into its diagonal part *D* and its
    strictly off-diagonal part *R = A − D*, then iterates:

        x^{k+1} = D^{-1} (b − R x^k)

    Convergence is guaranteed when *A* is strictly diagonally dominant.
    For general SPD matrices convergence may be slow or may not occur.

    The off-diagonal sparse matrix ``A_offdiag`` is computed **once** before
    the loop and reused at every iteration to avoid repeated sparse copies.

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
        """Run the Jacobi iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square SPD (or strictly diagonally
                dominant) coefficient matrix of shape ``(n, n)`` in CSR
                format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields:

            * ``solution`` – approximate solution vector.
            * ``iterations`` – number of iterations performed.
            * ``elapsed_seconds`` – wall-clock solve time.
            * ``relative_residual`` – ``‖Ax − b‖₂ / ‖b‖₂`` at termination.
            * ``converged`` – ``True`` if tolerance was reached.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, or any
                diagonal entry of *A* is (numerically) zero.
        """
        self._validate_inputs(A, b)

        diag = A.diagonal()
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("Jacobi requires non-zero diagonal")

        n = b.shape[0]
        x = np.zeros(n)  # initial iterate x^0 = 0

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        # Pre-compute the off-diagonal sparse matrix R = A − D once.
        # Reusing it avoids one sparse copy per iteration.
        A_offdiag = A.copy()
        A_offdiag.setdiag(0)

        start = perf_counter()

        for k in range(self.max_iter):
            rel_res = np.linalg.norm(A @ x - b) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            # Jacobi update: x^{k+1} = (b − R x^k) / d
            # Written in-place to avoid allocating a new array each iteration.
            x[:] = (b - A_offdiag @ x) / diag

        rel_res = np.linalg.norm(A @ x - b) / norm_b  # final residual check
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


# ---------------------------------------------------------------------------
# Gauss-Seidel
# ---------------------------------------------------------------------------

class GaussSeidelSolver(IterativeSolver):
    """Gauss-Seidel stationary iteration for sparse linear systems.

    The Gauss-Seidel method updates each component *x_i* using the **most
    recent values** of all other components:

        x_i^{k+1} = (b_i − Σ_{j≠i} a_{ij} x_j) / a_{ii}

    where updated values (``j < i``) are used as soon as they are available.
    This in-place sequential update typically converges faster than Jacobi for
    SPD matrices.

    The inner loop accesses the CSR data arrays (``data``, ``indices``,
    ``indptr``) directly to compute the row dot-product without materialising
    dense temporary vectors.

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
        """Run the Gauss-Seidel iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square SPD (or strictly diagonally
                dominant) coefficient matrix of shape ``(n, n)`` in CSR
                format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields:

            * ``solution`` – approximate solution vector.
            * ``iterations`` – number of iterations performed.
            * ``elapsed_seconds`` – wall-clock solve time.
            * ``relative_residual`` – ``‖Ax − b‖₂ / ‖b‖₂`` at termination.
            * ``converged`` – ``True`` if tolerance was reached.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, or any
                diagonal entry of *A* is (numerically) zero.
        """
        self._validate_inputs(A, b)

        diag = A.diagonal()
        if np.any(np.isclose(diag, 0.0)):
            raise ValueError("Gauss-Seidel requires non-zero diagonal")

        n = A.shape[0]
        x = np.zeros(n)  # initial iterate x^0 = 0

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()

        for k in range(self.max_iter):
            rel_res = np.linalg.norm(A @ x - b) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            for i in range(n):
                # Retrieve the non-zero entries for row i from CSR storage.
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                data = A.data[row_start:row_end]
                indices = A.indices[row_start:row_end]

                # Vectorised dot product over the full row i, then subtract
                # the diagonal contribution to obtain Σ_{j≠i} a_{ij} x_j.
                sum_Ax = np.dot(data, x[indices]) - diag[i] * x[i]

                # Gauss-Seidel component update (uses the latest x values).
                x[i] = (b[i] - sum_Ax) / diag[i]

        rel_res = np.linalg.norm(A @ x - b) / norm_b  # final residual check
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


# ---------------------------------------------------------------------------
# Gradient (steepest descent)
# ---------------------------------------------------------------------------

class GradientSolver(IterativeSolver):
    """Steepest-descent (gradient) method for SPD linear systems.

    At each step the optimal step size along the current residual direction is:

        α_k = ‖r_k‖₂² / (r_k^T A r_k)

    followed by the updates:

        x^{k+1} = x^k + α_k r_k
        r^{k+1} = r^k − α_k A r_k

    The residual is propagated analytically (without recomputing ``Ax``) so
    each iteration requires exactly **one** sparse matrix-vector product.

    Note:
        This method requires *A* to be **symmetric**. Positive definiteness is
        verified implicitly: a non-positive denominator ``r^T A r ≤ 0`` raises
        a :exc:`ValueError`.

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
        """Run the steepest-descent iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square, **symmetric** and positive
                definite coefficient matrix of shape ``(n, n)`` in CSR format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields:

            * ``solution`` – approximate solution vector.
            * ``iterations`` – number of iterations performed.
            * ``elapsed_seconds`` – wall-clock solve time.
            * ``relative_residual`` – ``‖Ax − b‖₂ / ‖b‖₂`` at termination.
            * ``converged`` – ``True`` if tolerance was reached.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, *A* is not
                symmetric (checked with ``max |A − Aᵀ| > 1e-10``), or the
                matrix is not positive definite (``r^T A r ≤ 0``).
        """
        self._validate_inputs(A, b)

        if np.abs(A - A.T).max() > 1e-10:
            raise ValueError("Gradient requires symmetric matrix")

        n = b.shape[0]
        x = np.zeros(n)   # initial iterate x^0 = 0
        r = b.copy()       # initial residual r^0 = b − A·0 = b

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()

        for k in range(self.max_iter):
            # Use the maintained residual r to avoid recomputing A @ x.
            rel_res = np.linalg.norm(r) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            Ar = A @ r  # single sparse mat-vec per iteration
            denom = np.dot(r, Ar)  # r^T A r — must be > 0 for SPD matrices
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")

            alpha = np.dot(r, r) / denom  # optimal step length α_k = ‖r‖² / (r^T A r)

            # In-place updates to avoid allocating new arrays.
            x += alpha * r
            r -= alpha * Ar

        rel_res = self._relative_residual(A, x, b)  # final residual check
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


# ---------------------------------------------------------------------------
# Conjugate Gradient
# ---------------------------------------------------------------------------

class ConjugateGradientSolver(IterativeSolver):
    """Conjugate Gradient (CG) method for SPD linear systems.

    The CG method selects search directions *p_k* that are *A*-conjugate
    (mutually orthogonal with respect to the *A*-inner product), ensuring
    finite-step convergence in exact arithmetic. In floating-point arithmetic
    convergence is superlinear and generally much faster than the gradient
    method.

    Iteration:

        α_k = ‖r_k‖₂² / (p_k^T A p_k)
        x^{k+1} = x^k + α_k p_k
        r^{k+1} = r^k − α_k A p_k
        β_k     = ‖r^{k+1}‖₂² / ‖r_k‖₂²
        p^{k+1} = r^{k+1} + β_k p_k

    The residual is propagated analytically so each iteration costs a single
    sparse matrix-vector product and three dot products.

    Note:
        This method requires *A* to be **symmetric positive definite**. The
        symmetry check uses ``max |A − Aᵀ| > 1e-10``; positive definiteness
        is verified implicitly via the sign of the denominator ``p^T A p``.

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
        """Run the Conjugate Gradient iteration to solve *Ax = b*.

        Args:
            A (scipy.sparse.csr_matrix): Square, **symmetric positive
                definite** coefficient matrix of shape ``(n, n)`` in CSR
                format.
            b (numpy.ndarray): Right-hand-side vector of shape ``(n,)``.

        Returns:
            SolverResult: Dataclass with fields:

            * ``solution`` – approximate solution vector.
            * ``iterations`` – number of iterations performed.
            * ``elapsed_seconds`` – wall-clock solve time.
            * ``relative_residual`` – ``‖Ax − b‖₂ / ‖b‖₂`` at termination.
            * ``converged`` – ``True`` if tolerance was reached.

        Raises:
            ValueError: If *A* is not square, dimensions mismatch, *A* is not
                symmetric (checked with ``max |A − Aᵀ| > 1e-10``), or the
                matrix is not positive definite (``p^T A p ≤ 0``).
        """
        self._validate_inputs(A, b)

        if np.abs(A - A.T).max() > 1e-10:
            raise ValueError("Conjugate Gradient requires symmetric matrix")

        n = b.shape[0]
        x = np.zeros(n)  # initial iterate x^0 = 0
        r = b.copy()     # initial residual r^0 = b − A·0 = b
        p = r.copy()     # initial search direction p^0 = r^0

        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            norm_b = 1.0  # fall back to absolute residual

        start = perf_counter()

        for k in range(self.max_iter):
            # Use the maintained residual r to avoid recomputing A @ x.
            rel_res = np.linalg.norm(r) / norm_b
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            Ap = A @ p  # single sparse mat-vec per iteration
            denom = np.dot(p, Ap)  # p^T A p — must be > 0 for SPD matrices
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")

            rr_old = np.dot(r, r)           # ‖r_k‖₂² before the update
            alpha = rr_old / denom           # step length α_k

            # In-place updates to avoid allocating new arrays.
            x += alpha * p
            r -= alpha * Ap

            rr_new = np.dot(r, r)            # ‖r_{k+1}‖₂² after the update
            beta = rr_new / rr_old           # Fletcher–Reeves coefficient β_k

            # Update the search direction p in-place: p ← r + β p
            p *= beta
            p += r

        rel_res = self._relative_residual(A, x, b)  # final residual check
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)
