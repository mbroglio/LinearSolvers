"""
Script to run complete tests on all matrices with different tolerances.
Saves results in CSV format for analysis and plot generation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from linear_solvers.numba import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
)


def run_complete_tests():
    """Run complete tests according to project specifications."""

    # Test matrices
    matrix_files = [
        "spa1.mtx",
        "spa2.mtx",
        "vem1.mtx",
        "vem2.mtx",
    ]

    # Tolerances to test (from specification)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]

    # Parameters
    max_iter = 20000
    data_dir = Path("data")

    # Results for each matrix
    all_results = {}

    print("=" * 100)
    print("TEST SUITE - Linear Solvers Validation")
    print("=" * 100)
    print(f"Max iterations: {max_iter}")
    print(f"Tolerances: {tolerances}")
    print("=" * 100)
    print()

    # Iterate over each matrix
    for matrix_file in matrix_files:
        matrix_path = data_dir / matrix_file

        if not matrix_path.exists():
            print(f"WARNING: Matrix {matrix_file} not found, skipping...")
            continue

        print(f"\n{'='*100}")
        print(f"Testing matrix: {matrix_file}")
        print(f"{'='*100}")

        # Load matrix
        A = mmread(matrix_path).tocsr()
        print(f"Matrix shape: {A.shape}")
        num_elements = A.shape[0] * A.shape[1]
        print(f"Matrix density: {A.nnz / num_elements * 100:.2f}%")

        # Create exact solution and right-hand side (Steps 1 and 2 from specification)
        x_exact = np.ones(A.shape[0])
        b = A @ x_exact

        # Results for this matrix
        matrix_results = []

        # Iterate over each tolerance
        for tol in tolerances:
            print(f"\n{'-'*100}")
            print(f"Tolerance: {tol:.0e}")
            print(f"{'-'*100}")
            print(
                f"{'Method':<25} {'Conv':<8} {'RelErr':<14} {'RelRes':<14} {'Iters':<10} {'Time[s]':<12}"
            )
            print(f"{'-'*100}")

            # Define solvers
            solvers = [
                JacobiSolver(tol, max_iter),
                GaussSeidelSolver(tol, max_iter),
                GradientSolver(tol, max_iter),
                ConjugateGradientSolver(tol, max_iter),
            ]

            # Test each solver
            for solver in solvers:
                try:
                    # Solve the system (Step 3)
                    result = solver.solve(A, b)

                    # Calculate relative error (Step 4)
                    rel_error = np.linalg.norm(
                        result.solution - x_exact
                    ) / np.linalg.norm(x_exact)

                    # Convergence
                    conv_str = "Yes" if result.converged else "No"

                    # Print results
                    print(
                        f"{solver.name:<25} {conv_str:<8} {rel_error:<14.6e} "
                        f"{result.relative_residual:<14.6e} {result.iterations:<10} "
                        f"{result.elapsed_seconds:<12.6f}"
                    )

                    # Save results
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

                    # Save error result
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

    # Convert to DataFrame for analysis
    df = pd.DataFrame([item for results in all_results.values() for item in results])

    # Save results
    save_results(df)

    print("\n" + "=" * 100)
    print("Tests completed! Results saved in 'results/' directory")
    print("Run 'python plot_results.py' to generate plots.")
    print("=" * 100)

    return df


def save_results(df: pd.DataFrame):
    """Save results in CSV and Excel format."""

    # Create output directory
    output_dir = Path("runner/results/numba")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("Saving results...")
    print("=" * 100)

    # 1. Summary table for each matrix and tolerance
    print("\nGenerating summary table...")
    summary_table = df.pivot_table(
        index=["matrix", "tolerance"],
        columns="method",
        values=["iterations", "time", "converged"],
        aggfunc="first",
    )
    summary_table.to_csv(output_dir / "summary_table.csv")
    print(f"✓ Table saved in {output_dir / 'summary_table.csv'}")

    # 2. Detailed table for each matrix
    print("\nGenerating detailed tables...")
    matrices = df["matrix"].unique()

    for matrix in matrices:
        matrix_data = df[df["matrix"] == matrix].copy()
        matrix_data = matrix_data.sort_values(["tolerance", "method"])

        # Format for report
        report_table = matrix_data[
            ["method", "tolerance", "converged", "relative_error", "iterations", "time"]
        ].copy()

        report_table.to_csv(
            output_dir / f"detailed_report_{matrix.replace('.mtx', '')}.csv",
            index=False,
        )

    print(f"✓ Detailed tables saved in {output_dir}")

    # 3. Print final statistics
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
    # Install pandas if needed
    try:
        import pandas
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        print("Packages installed successfully!")

    # Run tests
    df = run_complete_tests()

    print("\n✓ All tests completed successfully!")
    print(f"✓ Results available in 'results/' directory")
    print(f"✓ Run 'python plot_results.py' to generate plots")
