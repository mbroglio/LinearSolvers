"""Command-line entry point for quick validation of iterative solvers.

This script loads a sparse matrix from a Matrix Market (``.mtx``) file,
constructs a synthetic right-hand side ``b = A @ x_exact`` with
``x_exact = [1, 1, ..., 1]``, runs all four baseline solvers, and prints a
formatted comparison table with convergence status, relative error, relative
residual, iteration count, and elapsed time.

Usage::

    python main.py path/to/matrix.mtx [--tol TOL] [--max-iter MAX_ITER]

Arguments:
    matrix (Path): Path to a Matrix Market ``.mtx`` file. The matrix is loaded
        and converted to CSR format via :func:`scipy.io.mmread`.
    --tol (float): Convergence tolerance (default: ``1e-10``).
    --max-iter (int): Maximum number of iterations (default: ``20000``).

Output columns:
    Method      – solver name.
    Conv        – ``Yes`` if the solver converged, ``No`` otherwise.
    RelErr      – relative error ``‖x − x_exact‖₂ / ‖x_exact‖₂``.
    RelRes      – relative residual ``‖Ax − b‖₂ / ‖b‖₂`` at termination.
    Iters       – number of iterations performed.
    Time[s]     – wall-clock time in seconds.

Example::

    python main.py data/spa1.mtx --tol 1e-8
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import mmread

from linear_solvers.baseline import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)


def main() -> None:
    """Parse arguments, run all solvers, and print the results table."""
    parser = argparse.ArgumentParser(description="Validate iterative solvers")
    parser.add_argument("matrix", type=Path, help="Path to .mtx file")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance")
    parser.add_argument("--max-iter", type=int, default=20000, help="Max iterations")
    args = parser.parse_args()

    # Load the sparse matrix and convert to CSR format for efficient row access.
    A = mmread(args.matrix).tocsr()

    # Synthetic validation: x_exact = [1, …, 1], b = A @ x_exact.
    # The relative error ‖x − x_exact‖ / ‖x_exact‖ measures solution accuracy.
    x_exact = np.ones(A.shape[0])
    b = A @ x_exact

    solvers = [
        JacobiSolver(args.tol, args.max_iter),
        GaussSeidelSolver(args.tol, args.max_iter),
        GradientSolver(args.tol, args.max_iter),
        ConjugateGradientSolver(args.tol, args.max_iter),
    ]

    print(f"Matrix: {args.matrix.name}, shape: {A.shape}, tol: {args.tol}")
    print(f"Validation: x_exact = [1,1,...,1], b = A @ x_exact\n")
    print(
        f"{'Method':<22} {'Conv':<7} {'RelErr':<12} {'RelRes':<12} {'Iters':<8} {'Time[s]':<10}"
    )
    print("-" * 80)

    for solver in solvers:
        try:
            result = solver.solve(A, b)
            # Relative error: ‖x − x_exact‖₂ / ‖x_exact‖₂
            err = np.linalg.norm(result.solution - x_exact) / np.linalg.norm(x_exact)
            conv = "Yes" if result.converged else "No"
            print(
                f"{solver.name:<22} {conv:<7} {err:<12.6e} {result.relative_residual:<12.6e} "
                f"{result.iterations:<8} {result.elapsed_seconds:<10.6f}"
            )
        except Exception as e:
            print(
                f"{solver.name:<22} No      -            -            -        -          ({str(e)})"
            )


if __name__ == "__main__":
    main()
