"""
Script to generate plots from pre-calculated results.
Loads data from CSV and generates customizable plots without re-running tests.
All plots are saved in SVG format.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

BASELINE_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "baseline"

# Fixed color mapping to avoid confusion between methods in different plots
METHOD_COLORS = {
    "Jacobi": "tab:blue",
    "Gauss-Seidel": "tab:orange",
    "Gradient": "tab:green",
    "Conjugate Gradient": "tab:red",
}


def load_results():
    """Load results from CSV files."""
    results_dir = BASELINE_RESULTS_DIR

    # Load all detailed reports
    all_data = []
    for csv_file in results_dir.glob("detailed_report_*.csv"):
        df = pd.read_csv(csv_file)

        # Extract matrix name from filename
        # e.g.: detailed_report_spa1.csv -> spa1.mtx
        matrix_name = csv_file.stem.replace("detailed_report_", "") + ".mtx"
        df["matrix"] = matrix_name

        all_data.append(df)

    if not all_data:
        print("ERROR: No result files found in results/")
        print("Run first: python run_tests.py")
        return None

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    print(f"OK: Loaded {len(df)} results from {len(all_data)} files")
    return df


def plot_iterations_vs_tolerance(df, output_dir="results", show=False):
    """Generate iterations vs tolerance plots for each matrix."""
    matrices = df["matrix"].unique()

    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        methods = matrix_data["method"].unique()
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")

            converged_data = method_data[method_data["converged"] == True]

            if len(converged_data) > 0:
                ax.plot(
                    converged_data["tolerance"],
                    converged_data["iterations"],
                    marker="o",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Tolerance", fontsize=12)
        ax.set_ylabel("Iterations", fontsize=12)
        ax.set_title(
            f"Iterations vs Tolerance - {matrix}", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = (
            Path(output_dir)
            / f"iterations_vs_tolerance_{matrix.replace('.mtx', '')}.svg"
        )
        plt.savefig(output_path, format="svg")
        print(f"OK: Salvato: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_iterations_vs_tolerance_no_gradient(df, output_dir="results", show=False):
    """Generate iterations vs tolerance plots for spa1 and spa2 (without gradient)."""
    matrices = df["matrix"].unique()
    spa_matrices = [m for m in matrices if "spa1" in m or "spa2" in m]

    for matrix in spa_matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        methods = matrix_data["method"].unique()
        for method in methods:
            # Skip only gradient method (not conjugate gradient)
            if method.lower() == "gradient":
                continue

            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")

            converged_data = method_data[method_data["converged"] == True]

            if len(converged_data) > 0:
                ax.plot(
                    converged_data["tolerance"],
                    converged_data["iterations"],
                    marker="o",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Tolerance", fontsize=12)
        ax.set_ylabel("Iterations", fontsize=12)
        ax.set_title(
            f"Iterations vs Tolerance - {matrix} (no gradient)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = (
            Path(output_dir)
            / f"iterations_vs_tolerance_{matrix.replace('.mtx', '')}_no_gradient.svg"
        )
        plt.savefig(output_path, format="svg")
        print(f"OK: Salvato: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_time_vs_tolerance(df, output_dir="results", show=False):
    """Generate time vs tolerance plots for each matrix."""
    matrices = df["matrix"].unique()

    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        methods = matrix_data["method"].unique()
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")

            converged_data = method_data[method_data["converged"] == True]

            if len(converged_data) > 0:
                ax.plot(
                    converged_data["tolerance"],
                    converged_data["time"],
                    marker="s",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Tolerance", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title(
            f"Execution Time vs Tolerance - {matrix}", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = (
            Path(output_dir) / f"time_vs_tolerance_{matrix.replace('.mtx', '')}.svg"
        )
        plt.savefig(output_path, format="svg")
        print(f"OK: Salvato: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_error_vs_tolerance(df, output_dir="results", show=False):
    """Generate relative error vs tolerance plots for each matrix."""
    matrices = df["matrix"].unique()

    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        methods = matrix_data["method"].unique()
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")

            converged_data = method_data[
                (method_data["converged"] == True)
                & (~method_data["relative_error"].isna())
            ]

            if len(converged_data) > 0:
                ax.plot(
                    converged_data["tolerance"],
                    converged_data["relative_error"],
                    marker="^",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tolerance", fontsize=12)
        ax.set_ylabel("Relative Error", fontsize=12)
        ax.set_title(
            f"Relative Error vs Tolerance - {matrix}", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = (
            Path(output_dir) / f"error_vs_tolerance_{matrix.replace('.mtx', '')}.svg"
        )
        plt.savefig(output_path, format="svg")
        print(f"OK: Salvato: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_comparative(df, output_dir="results", show=False):
    """Generate comparative plot with all matrices."""
    matrices = df["matrix"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, matrix in enumerate(matrices):
        if idx >= len(axes):
            break

        ax = axes[idx]
        matrix_data = df[df["matrix"] == matrix]

        methods = matrix_data["method"].unique()
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")

            converged_data = method_data[method_data["converged"] == True]

            if len(converged_data) > 0:
                ax.plot(
                    converged_data["tolerance"],
                    converged_data["iterations"],
                    marker="o",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Tolerance", fontsize=10)
        ax.set_ylabel("Iterations", fontsize=10)
        ax.set_title(f"{matrix}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "comparative_all_matrices.svg"
    plt.savefig(output_path, format="svg")
    print(f"OK: Salvato: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_method_comparison(df, output_dir="results", show=False):
    """Direct comparison between methods for each matrix."""
    matrices = df["matrix"].unique()

    for matrix in matrices:
        matrix_data = df[df["matrix"] == matrix]

        # Subplot with 3 plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Iterazioni
        ax = axes[0]
        methods = matrix_data["method"].unique()
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")
            converged = method_data[method_data["converged"] == True]
            if len(converged) > 0:
                ax.plot(
                    converged["tolerance"],
                    converged["iterations"],
                    marker="o",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Tolerance", fontsize=11)
        ax.set_ylabel("Iterations", fontsize=11)
        ax.set_title("Iterations", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Time
        ax = axes[1]
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")
            converged = method_data[method_data["converged"] == True]
            if len(converged) > 0:
                ax.plot(
                    converged["tolerance"],
                    converged["time"],
                    marker="s",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Tolerance", fontsize=11)
        ax.set_ylabel("Time (s)", fontsize=11)
        ax.set_title("Execution Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Error
        ax = axes[2]
        for method in methods:
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")
            converged = method_data[
                (method_data["converged"] == True)
                & (~method_data["relative_error"].isna())
            ]
            if len(converged) > 0:
                ax.plot(
                    converged["tolerance"],
                    converged["relative_error"],
                    marker="^",
                    label=method,
                    color=METHOD_COLORS.get(method),
                    linewidth=2,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tolerance", fontsize=11)
        ax.set_ylabel("Relative Error", fontsize=11)
        ax.set_title("Relative Error", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Method Comparison - {matrix}", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        output_path = (
            Path(output_dir) / f"method_comparison_{matrix.replace('.mtx', '')}.svg"
        )
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        print(f"OK: Salvato: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for matrix in df["matrix"].unique():
        matrix_data = df[df["matrix"] == matrix]
        print(f"\n{matrix}:")

        for method in matrix_data["method"].unique():
            method_data = matrix_data[matrix_data["method"] == method]

            # Find best result (tightest tolerance)
            best = method_data.loc[method_data["tolerance"].idxmin()]

            print(
                f"  {method:25s}: {best['iterations']:5.0f} iter, "
                f"{best['time']:8.4f}s @ tol={best['tolerance']:.0e}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from pre-calculated test results"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=[
            "iterations",
            "iterations-no-gradient",
            "time",
            "error",
            "comparative",
            "comparison",
            "all",
        ],
        default=["all"],
        help="Types of plots to generate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASELINE_RESULTS_DIR),
        help="Output directory (default: runner/results/baseline)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show plots instead of just saving them"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PLOT GENERATOR")
    print("=" * 80)

    # Load data
    df = load_results()
    if df is None:
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating SVG plots...")
    print(f"Output: {output_dir}/")

    # Determine which plots to generate
    plot_types = args.plots
    if "all" in plot_types:
        plot_types = [
            "iterations",
            "iterations-no-gradient",
            "time",
            "error",
            "comparative",
            "comparison",
        ]

    # Generate requested plots
    if "iterations" in plot_types:
        print("\nIterations vs tolerance plots...")
        plot_iterations_vs_tolerance(df, args.output, args.show)

    if "iterations-no-gradient" in plot_types:
        print("\nIterations vs tolerance plots (spa1/spa2, no gradient)...")
        plot_iterations_vs_tolerance_no_gradient(df, args.output, args.show)

    if "time" in plot_types:
        print("\nTime vs tolerance plots...")
        plot_time_vs_tolerance(df, args.output, args.show)

    if "error" in plot_types:
        print("\nError vs tolerance plots...")
        plot_error_vs_tolerance(df, args.output, args.show)

    if "comparative" in plot_types:
        print("\nComparative plot...")
        plot_comparative(df, args.output, args.show)

    if "comparison" in plot_types:
        print("\nMethod comparison plots...")
        plot_method_comparison(df, args.output, args.show)

    # Print statistics
    print_summary(df)

    print("\n" + "=" * 80)
    print(f"DONE! SVG plots generated in: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
