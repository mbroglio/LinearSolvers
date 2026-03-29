import argparse
from pathlib import Path
import numpy as np
from src.linear_solvers import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
    load_matrix_market,
)


def main():
    parser = argparse.ArgumentParser(description="Validate iterative solvers")
    parser.add_argument("matrix", type=Path, help="Path to .mtx file")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance")
    parser.add_argument("--max-iter", type=int, default=20000, help="Max iterations")
    args = parser.parse_args()

    A = load_matrix_market(args.matrix)
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
    print(f"{'Method':<22} {'Conv':<7} {'RelErr':<12} {'RelRes':<12} {'Iters':<8} {'Time[s]':<10}")
    print("-" * 80)

    for solver in solvers:
        try:
            result = solver.solve(A, b)
            err = np.linalg.norm(result.solution - x_exact) / np.linalg.norm(x_exact)
            conv = "Yes" if result.converged else "No"
            print(f"{solver.name:<22} {conv:<7} {err:<12.6e} {result.relative_residual:<12.6e} "
                  f"{result.iterations:<8} {result.elapsed_seconds:<10.6f}")
        except Exception as e:
            print(f"{solver.name:<22} No      -            -            -        -          ({str(e)})")


if __name__ == "__main__":
    main()
