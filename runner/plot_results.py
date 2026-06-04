"""runner.plot_results: Generate diagnostic plots from pre-computed benchmark data.

This script loads the CSV result files produced by :mod:`runner.run_baseline`
and generates a comprehensive set of SVG plots comparing the four iterative
solvers across all test matrices and tolerances.

All plots use a **fixed colour mapping** (defined in :data:`METHOD_COLORS`) to
ensure consistent visual identification of solvers across different chart types.

Available plot types (selectable via ``--plots``):
    ``iterations``
        Iterations vs. tolerance for each matrix (all methods, log-x axis).
    ``iterations-no-gradient``
        Same as above but restricted to ``spa1`` and ``spa2`` matrices and
        excluding the plain Gradient method (to improve y-axis readability when
        Jacobi/Gauss-Seidel iteration counts dominate).
    ``time``
        Wall-clock time (seconds) vs. tolerance for each matrix (log-x axis).
    ``error``
        Relative error vs. tolerance for each matrix (log-log axes).
    ``comparative``
        2×2 grid with iteration counts for all four matrices on a single figure.
    ``comparison``
        Three-panel figure (iterations, time, error) for each matrix.
    ``all`` *(default)*
        All of the above.

Output files:
    All plots are saved as SVG files in the directory specified by ``--output``
    (default: ``runner/results/baseline/``).

Usage::

    python runner/plot_results.py [--plots TYPE [TYPE ...]] [--output DIR] [--show]

Arguments:
    --plots (str, optional): One or more plot types to generate. Defaults to
        ``all``.
    --output (str, optional): Output directory for SVG files. Defaults to
        ``runner/results/baseline/``.
    --show (flag, optional): If set, display each figure interactively instead
        of (or in addition to) saving it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "baseline"

# Fixed colour mapping to ensure visual consistency across all plot types.
METHOD_COLORS: dict[str, str] = {
    "Jacobi": "tab:blue",
    "Gauss-Seidel": "tab:orange",
    "Gradient": "tab:green",
    "Conjugate Gradient": "tab:red",
}


def load_results() -> pd.DataFrame | None:
    """Load pre-computed benchmark results from CSV files.

    Scans ``runner/results/baseline/`` for files matching
    ``detailed_report_*.csv``, reads each one, infers the matrix name from the
    filename, and concatenates everything into a single DataFrame.

    Returns:
        pandas.DataFrame | None: Combined result table if at least one CSV file
        was found, ``None`` otherwise.  Columns match the output of
        :func:`runner.run_baseline.run_complete_tests`.
    """
    results_dir = BASELINE_RESULTS_DIR

    all_data = []
    for csv_file in results_dir.glob("detailed_report_*.csv"):
        df = pd.read_csv(csv_file)

        # Reconstruct the .mtx filename from the report filename, e.g.
        # "detailed_report_spa1.csv" → "spa1.mtx".
        matrix_name = csv_file.stem.replace("detailed_report_", "") + ".mtx"
        df["matrix"] = matrix_name

        all_data.append(df)

    if not all_data:
        print("ERROR: No result files found in results/")
        print("Run first: python run_tests.py")
        return None

    df = pd.concat(all_data, ignore_index=True)
    print(f"OK: Loaded {len(df)} results from {len(all_data)} files")
    return df


def plot_iterations_vs_tolerance(
    df: pd.DataFrame, output_dir: str = "results", show: bool = False
) -> None:
    """Plot iteration count vs. tolerance for each test matrix.

    Generates one SVG file per matrix. Only **converged** runs are plotted;
    diverged or failed runs are silently skipped.

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
        output_dir (str, optional): Directory where SVG files are saved.
            Defaults to ``"results"``.
        show (bool, optional): If ``True``, call :func:`matplotlib.pyplot.show`
            after saving each figure. Defaults to ``False``.
    """
    matrices = df["matrix"].unique()

    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        for method in matrix_data["method"].unique():
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


def plot_iterations_vs_tolerance_no_gradient(
    df: pd.DataFrame, output_dir: str = "results", show: bool = False
) -> None:
    """Plot iteration count vs. tolerance for spa1/spa2, excluding the Gradient method.

    This variant is useful to compare Jacobi, Gauss-Seidel, and Conjugate
    Gradient on the ``spa`` matrices without the Gradient method's large
    iteration counts compressing the y-axis.

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
        output_dir (str, optional): Directory where SVG files are saved.
            Defaults to ``"results"``.
        show (bool, optional): If ``True``, display each figure interactively.
            Defaults to ``False``.
    """
    matrices = df["matrix"].unique()
    spa_matrices = [m for m in matrices if "spa1" in m or "spa2" in m]

    for matrix in spa_matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        for method in matrix_data["method"].unique():
            # Exclude only the plain steepest-descent method, not Conjugate Gradient.
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


def plot_time_vs_tolerance(
    df: pd.DataFrame, output_dir: str = "results", show: bool = False
) -> None:
    """Plot wall-clock solve time vs. tolerance for each test matrix.

    Generates one SVG file per matrix. Only **converged** runs are plotted.

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
        output_dir (str, optional): Directory where SVG files are saved.
            Defaults to ``"results"``.
        show (bool, optional): If ``True``, display each figure interactively.
            Defaults to ``False``.
    """
    matrices = df["matrix"].unique()

    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        for method in matrix_data["method"].unique():
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


def plot_error_vs_tolerance(
    df: pd.DataFrame, output_dir: str = "results", show: bool = False
) -> None:
    """Plot relative solution error vs. tolerance for each test matrix (log-log).

    Generates one SVG file per matrix. Only **converged** runs with a valid
    (non-NaN) relative error are plotted.

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
        output_dir (str, optional): Directory where SVG files are saved.
            Defaults to ``"results"``.
        show (bool, optional): If ``True``, display each figure interactively.
            Defaults to ``False``.
    """
    matrices = df["matrix"].unique()

    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df["matrix"] == matrix]

        for method in matrix_data["method"].unique():
            method_data = matrix_data[matrix_data["method"] == method]
            method_data = method_data.sort_values("tolerance")

            # Filter to converged rows with a finite relative error.
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


def plot_comparative(
    df: pd.DataFrame, output_dir: str = "results", show: bool = False
) -> None:
    """Plot a 2×2 grid of iteration-count curves for all four test matrices.

    Each sub-plot shows iterations vs. tolerance for a single matrix using the
    shared :data:`METHOD_COLORS` palette.  The combined figure provides a quick
    visual overview of solver behaviour across the entire test suite.

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
        output_dir (str, optional): Directory where the SVG file is saved.
            Defaults to ``"results"``.
        show (bool, optional): If ``True``, display the figure interactively.
            Defaults to ``False``.
    """
    matrices = df["matrix"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, matrix in enumerate(matrices):
        if idx >= len(axes):
            break

        ax = axes[idx]
        matrix_data = df[df["matrix"] == matrix]

        for method in matrix_data["method"].unique():
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


def plot_method_comparison(
    df: pd.DataFrame, output_dir: str = "results", show: bool = False
) -> None:
    """Generate a three-panel method comparison figure for each test matrix.

    Each figure contains three side-by-side panels:

    1. **Iterations** vs. tolerance.
    2. **Execution time** (seconds) vs. tolerance.
    3. **Relative error** vs. tolerance (log-log).

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
        output_dir (str, optional): Directory where SVG files are saved.
            Defaults to ``"results"``.
        show (bool, optional): If ``True``, display each figure interactively.
            Defaults to ``False``.
    """
    matrices = df["matrix"].unique()

    for matrix in matrices:
        matrix_data = df[df["matrix"] == matrix]
        methods = matrix_data["method"].unique()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- Panel 1: Iterations vs. tolerance ---
        ax = axes[0]
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

        # --- Panel 2: Wall-clock time vs. tolerance ---
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

        # --- Panel 3: Relative error vs. tolerance (log-log) ---
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


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of solver performance to stdout.

    For each ``(matrix, method)`` pair, reports the iteration count and
    wall-clock time at the **tightest** tolerance available in the dataset.

    Args:
        df (pandas.DataFrame): Result table as returned by :func:`load_results`.
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for matrix in df["matrix"].unique():
        matrix_data = df[df["matrix"] == matrix]
        print(f"\n{matrix}:")

        for method in matrix_data["method"].unique():
            method_data = matrix_data[matrix_data["method"] == method]

            # Select the row corresponding to the tightest (smallest) tolerance.
            best = method_data.loc[method_data["tolerance"].idxmin()]

            print(
                f"  {method:25s}: {best['iterations']:5.0f} iter, "
                f"{best['time']:8.4f}s @ tol={best['tolerance']:.0e}"
            )


def main() -> None:
    """Parse command-line arguments and dispatch plot generation.

    Reads result CSVs via :func:`load_results`, then calls the requested
    plotting functions in order.
    """
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

    df = load_results()
    if df is None:
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating SVG plots...")
    print(f"Output: {output_dir}/")

    # Expand "all" into the full list of plot types.
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

    print_summary(df)

    print("\n" + "=" * 80)
    print(f"DONE! SVG plots generated in: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
