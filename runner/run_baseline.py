"""runner.run_baseline: Benchmark script for the baseline iterative solvers.

This script runs the full test suite defined in the project specification:

* **Matrices**: ``spa1.mtx``, ``spa2.mtx``, ``vem1.mtx``, ``vem2.mtx``
  (loaded from the ``data/`` directory relative to the project root).
* **Tolerances**: ``1e-4``, ``1e-6``, ``1e-8``, ``1e-10``.
* **Solvers**: :class:`~linear_solvers.baseline.JacobiSolver`,
  :class:`~linear_solvers.baseline.GaussSeidelSolver`,
  :class:`~linear_solvers.baseline.GradientSolver`,
  :class:`~linear_solvers.baseline.ConjugateGradientSolver`.

For each ``(matrix, tolerance)`` pair every solver attempts to solve the
synthetic system ``Ax = b`` where ``b = A @ ones(n)`` (so the exact solution
is ``x_exact = [1, …, 1]``).  Results are collected in a
:class:`pandas.DataFrame` and saved as CSV files under
``runner/results/baseline/``.

Output files:
    runner/results/baseline/summary_table.csv
        Pivot table with iterations, time, and convergence for each
        ``(matrix, tolerance, method)`` triple.
    runner/results/baseline/detailed_report_<matrix>.csv
        Per-matrix detailed table with relative error, relative residual,
        iterations, and time for every ``(method, tolerance)`` pair.

Usage (from the project root)::

    python -m runner.run_baseline
    # or
    python runner/run_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

from linear_solvers.baseline import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)


def run_complete_tests() -> pd.DataFrame:
    """Run the full benchmark suite on all matrices and tolerances.

    Iterates over every combination of test matrix and tolerance, instantiates
    the four baseline solvers, calls :meth:`~linear_solvers.baseline.IterativeSolver.solve`,
    and records the results.  Matrices that are not found on disk are skipped
    with a warning.

    Returns:
        pandas.DataFrame: A flat table with one row per
        ``(matrix, method, tolerance)`` triple and the following columns:

        * ``matrix`` (str) – filename of the test matrix.
        * ``method`` (str) – solver name.
        * ``tolerance`` (float) – convergence tolerance used.
        * ``converged`` (bool) – whether the solver reached the tolerance.
        * ``relative_error`` (float) – ``‖x − x_exact‖₂ / ‖x_exact‖₂``,
          or ``NaN`` on solver failure.
        * ``relative_residual`` (float) – ``‖Ax − b‖₂ / ‖b‖₂`` at
          termination, or ``NaN`` on solver failure.
        * ``iterations`` (int) – number of iterations performed (``0`` on
          solver failure).
        * ``time`` (float) – wall-clock solve time in seconds (``0.0`` on
          solver failure).
    """
    # Test matrices (must reside in data/ relative to the project root).
    matrix_files = [
        "spa1.mtx",
        "spa2.mtx",
        "vem1.mtx",
        "vem2.mtx",
    ]

    # Tolerances from the project specification.
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]

    max_iter = 20000
    data_dir = Path("data")

    # Accumulate results grouped by matrix name.
    all_results: dict[str, list[dict]] = {}

    print("=" * 100)
    print("TEST SUITE - Linear Solvers Validation")
    print("=" * 100)
    print(f"Max iterations: {max_iter}")
    print(f"Tolerances: {tolerances}")
    print("=" * 100)
    print()

    for matrix_file in matrix_files:
        matrix_path = data_dir / matrix_file

        if not matrix_path.exists():
            print(f"WARNING: Matrix {matrix_file} not found, skipping...")
            continue

        print(f"\n{'='*100}")
        print(f"Testing matrix: {matrix_file}")
        print(f"{'='*100}")

        # Load matrix and convert to CSR format.
        A = mmread(matrix_path).tocsr()
        print(f"Matrix shape: {A.shape}")
        num_elements = A.shape[0] * A.shape[1]
        print(f"Matrix density: {A.nnz / num_elements * 100:.2f}%")

        # Synthetic exact solution: x_exact = [1, …, 1], b = A @ x_exact.
        x_exact = np.ones(A.shape[0])
        b = A @ x_exact

        matrix_results: list[dict] = []

        for tol in tolerances:
            print(f"\n{'-'*100}")
            print(f"Tolerance: {tol:.0e}")
            print(f"{'-'*100}")
            print(
                f"{'Method':<25} {'Conv':<8} {'RelErr':<14} {'RelRes':<14} {'Iters':<10} {'Time[s]':<12}"
            )
            print(f"{'-'*100}")

            solvers = [
                JacobiSolver(tol, max_iter),
                GaussSeidelSolver(tol, max_iter),
                GradientSolver(tol, max_iter),
                ConjugateGradientSolver(tol, max_iter),
            ]

            for solver in solvers:
                try:
                    # Solve Ax = b (Step 3 of the project specification).
                    result = solver.solve(A, b)

                    # Relative error ‖x − x_exact‖₂ / ‖x_exact‖₂ (Step 4).
                    rel_error = np.linalg.norm(
                        result.solution - x_exact
                    ) / np.linalg.norm(x_exact)

                    conv_str = "Yes" if result.converged else "No"
                    print(
                        f"{solver.name:<25} {conv_str:<8} {rel_error:<14.6e} "
                        f"{result.relative_residual:<14.6e} {result.iterations:<10} "
                        f"{result.elapsed_seconds:<12.6f}"
                    )

                    matrix_results.append(
                        {
                            "matrix": matrix_file,
                            "method": solver.name,
                            "tolerance": tol,
                            "converged": result.converged,
                            "relative_error": rel_error,
                            "relative_residual": result.relative_residual,
                            "iterations": result.iterations,
                            "time": result.elapsed_seconds,
                        }
                    )

                except Exception as e:
                    print(f"{solver.name:<25} No       ERROR: {str(e)}")

                    # Record a failure row so the CSV remains complete.
                    matrix_results.append(
                        {
                            "matrix": matrix_file,
                            "method": solver.name,
                            "tolerance": tol,
                            "converged": False,
                            "relative_error": np.nan,
                            "relative_residual": np.nan,
                            "iterations": 0,
                            "time": 0.0,
                        }
                    )

        all_results[matrix_file] = matrix_results

    # Flatten all per-matrix result lists into a single DataFrame.
    df = pd.DataFrame([item for results in all_results.values() for item in results])

    save_results(df)

    print("\n" + "=" * 100)
    print("Tests completed! Results saved in 'results/' directory")
    print("Run 'python plot_results.py' to generate plots.")
    print("=" * 100)

    return df


def save_results(df: pd.DataFrame) -> None:
    """Persist benchmark results to CSV files.

    Creates the output directory ``runner/results/baseline/`` if it does not
    exist, then writes:

    1. A **summary pivot table** (``summary_table.csv``) indexed by
       ``(matrix, tolerance)`` with columns for each solver's iterations,
       wall-clock time, and convergence flag.
    2. A **per-matrix detailed report** (``detailed_report_<matrix>.csv``)
       sorted by ``(tolerance, method)``, containing method name, tolerance,
       convergence flag, relative error, iteration count, and time.

    Finally, a convergence statistics overview is printed to stdout.

    Args:
        df (pandas.DataFrame): Flat result table as returned by
            :func:`run_complete_tests`.
    """
    output_dir = Path("runner/results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("Saving results...")
    print("=" * 100)

    # 1. Summary pivot table: rows = (matrix, tolerance), columns = methods.
    print("\nGenerating summary table...")
    summary_table = df.pivot_table(
        index=["matrix", "tolerance"],
        columns="method",
        values=["iterations", "time", "converged"],
        aggfunc="first",
    )
    summary_table.to_csv(output_dir / "summary_table.csv")
    print(f"✓ Table saved in {output_dir / 'summary_table.csv'}")

    # 2. Detailed per-matrix report for each test matrix.
    print("\nGenerating detailed tables...")
    matrices = df["matrix"].unique()

    for matrix in matrices:
        matrix_data = df[df["matrix"] == matrix].copy()
        matrix_data = matrix_data.sort_values(["tolerance", "method"])

        report_table = matrix_data[
            ["method", "tolerance", "converged", "relative_error", "iterations", "time"]
        ].copy()

        report_table.to_csv(
            output_dir / f"detailed_report_{matrix.replace('.mtx', '')}.csv",
            index=False,
        )

    print(f"✓ Detailed tables saved in {output_dir}")

    # 3. Convergence statistics summary printed to stdout.
    print("\n" + "=" * 100)
    print("FINAL STATISTICS")
    print("=" * 100)

    for matrix in matrices:
        print(f"\n{matrix}:")
        matrix_data = df[df["matrix"] == matrix]

        for method in matrix_data["method"].unique():
            method_data = matrix_data[matrix_data["method"] == method]
            converged_count = method_data["converged"].sum()
            total_count = len(method_data)
            print(f"  {method:25s}: {converged_count}/{total_count} converged")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    # Ensure pandas is available (it is listed in pyproject.toml dependencies).
    try:
        import pandas
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        print("Packages installed successfully!")

    df = run_complete_tests()

    print("\n✓ All tests completed successfully!")
    print(f"✓ Results available in 'results/' directory")
    print(f"✓ Run 'python plot_results.py' to generate plots")
