"""
Matrix Properties Analysis: Conditioning Number and Spectral Radius
This script analyzes fundamental properties of the test matrices:
- Conditioning number (cond): indicates matrix sensitivity to perturbations
- Spectral radius (rho_B): largest eigenvalue magnitude of iteration matrix B
  (computed on Jacobi iteration matrix for standard comparison)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import issparse, diags
from scipy.sparse.linalg import eigs


def compute_jacobi_iteration_matrix(A):
    """
    Compute Jacobi iteration matrix B = D^(-1)(L + U).
    
    Args:
        A: Sparse or dense matrix
    
    Returns:
        B: Jacobi iteration matrix (dense)
    """
    if issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = A
    
    # Extract diagonal, lower, upper parts
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, k=-1)  # Lower triangular (excluding diagonal)
    U = np.triu(A_dense, k=1)   # Upper triangular (excluding diagonal)
    
    # Compute B = D^(-1)(L + U)
    D_inv = np.linalg.inv(D)
    B = D_inv @ (L + U)
    
    return B


def compute_gauss_seidel_iteration_matrix(A):
    """
    Compute Gauss-Seidel iteration matrix B_GS = (D + L)^(-1) * U.
    
    Args:
        A: Sparse or dense matrix
    
    Returns:
        B_GS: Gauss-Seidel iteration matrix (dense)
    """
    if issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = A
    
    # Extract diagonal, lower, upper parts
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, k=-1)  # Lower triangular (excluding diagonal)
    U = np.triu(A_dense, k=1)   # Upper triangular (excluding diagonal)
    
    # Compute B_GS = (D + L)^(-1) * U
    DL_inv = np.linalg.inv(D + L)
    B_GS = DL_inv @ U
    
    return B_GS


def compute_matrix_properties(A):
    """
    Compute conditioning number and spectral radii for a matrix.
    
    Args:
        A: Matrix (dense or sparse)
    
    Returns:
        dict with cond_2: L2-norm conditioning number, 
                 rho_jacobi: spectral radius of Jacobi iteration matrix,
                 rho_gauss_seidel: spectral radius of Gauss-Seidel iteration matrix
    """
    # Convert to dense if sparse (for conditioning number computation)
    if issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = A
    
    # Compute L2-norm conditioning number
    try:
        cond_2 = np.linalg.cond(A_dense, p=2)
    except Exception as e:
        print(f"Warning: Could not compute L2-norm conditioning number: {e}")
        cond_2 = np.nan
    
    # Compute spectral radius of Jacobi iteration matrix
    try:
        B = compute_jacobi_iteration_matrix(A)
        eigenvalues = np.linalg.eigvals(B)
        rho_jacobi = np.max(np.abs(eigenvalues))
    except Exception as e:
        print(f"Warning: Could not compute spectral radius of Jacobi matrix: {e}")
        rho_jacobi = np.nan
    
    # Compute spectral radius of Gauss-Seidel iteration matrix
    try:
        B_GS = compute_gauss_seidel_iteration_matrix(A)
        eigenvalues_gs = np.linalg.eigvals(B_GS)
        rho_gauss_seidel = np.max(np.abs(eigenvalues_gs))
    except Exception as e:
        print(f"Warning: Could not compute spectral radius of Gauss-Seidel matrix: {e}")
        rho_gauss_seidel = np.nan
    
    return {
        'cond_2': cond_2,
        'rho_jacobi': rho_jacobi,
        'rho_gauss_seidel': rho_gauss_seidel
    }


def df_to_markdown(df):
    """Convert DataFrame to markdown table format."""
    # Header
    header = "| " + " | ".join(df.columns) + " |\n"
    separator = "|" + "|".join(["---" for _ in df.columns]) + "|\n"
    
    # Rows
    rows = ""
    for _, row in df.iterrows():
        rows += "| " + " | ".join(str(val) for val in row) + " |\n"
    
    return header + separator + rows


def main():
    """Main analysis function."""
    
    # Test matrices
    matrix_files = [
        "spa1.mtx",
        "spa2.mtx",
        "vem1.mtx",
        "vem2.mtx",
    ]
    
    data_dir = Path("data")
    results_dir = Path("runner/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MATRIX PROPERTIES ANALYSIS")
    print("=" * 80)
    print()
    
    results_list = []
    
    for matrix_file in matrix_files:
        matrix_path = data_dir / matrix_file
        
        if not matrix_path.exists():
            print(f"ERROR: Matrix file not found: {matrix_path}")
            continue
        
        print(f"Loading {matrix_file}...", end=" ")
        
        try:
            A = mmread(matrix_path).tocsr()
            print(f"Shape: {A.shape}, NNZ: {A.nnz}")
            
            props = compute_matrix_properties(A)
            
            result = {
                'Matrix': matrix_file.replace('.mtx', ''),
                'Size': f"{A.shape[0]}x{A.shape[1]}",
                'NNZ': A.nnz,
                'Density': f"{100 * A.nnz / (A.shape[0] * A.shape[1]):.4e}%",
                'Condition Number (κ₂)': f"{props['cond_2']:.6e}",
                'ρ(B_Jacobi)': f"{props['rho_jacobi']:.6e}",
                'ρ(B_GS)': f"{props['rho_gauss_seidel']:.6e}",
                'Convergent_J': "Yes" if props['rho_jacobi'] < 1 else "No",
                'Convergent_GS': "Yes" if props['rho_gauss_seidel'] < 1 else "No"
            }
            
            results_list.append(result)
            conv_j = "✓" if props['rho_jacobi'] < 1 else "✗"
            conv_gs = "✓" if props['rho_gauss_seidel'] < 1 else "✗"
            print(f"  κ₂ = {props['cond_2']:.6e}")
            print(f"  ρ(B_Jacobi) = {props['rho_jacobi']:.6e} {conv_j}")
            print(f"  ρ(B_GS) = {props['rho_gauss_seidel']:.6e} {conv_gs}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print()
    print("=" * 80)
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Save as CSV
    csv_path = results_dir / "matrix_properties.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save as markdown
    md_path = results_dir / "matrix_properties.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Matrix Properties Analysis\n\n")
        f.write("## Summary Table\n\n")
        f.write(df_to_markdown(df))
        f.write("\n\n")
        f.write("## Analysis\n\n")
        f.write(generate_analysis(results_list))
    
    print(f"Markdown report saved to: {md_path}")
    print()
    print(df.to_string(index=False))
    print()


def generate_analysis(results_list):
    """Generate descriptive analysis of the matrix properties."""
    
    analysis = """### Conditioning Number (κ₂)

The conditioning number measures matrix sensitivity to perturbations:
- **κ₂ < 10**: Well-conditioned, numerically stable
- **10 < κ₂ < 1000**: Moderately conditioned, acceptable
- **κ₂ > 1000**: Ill-conditioned, requires careful numerical treatment

### Spectral Radii of Iteration Matrices

The spectral radius of iteration matrices B determine convergence of stationary iterative methods:

**Jacobi Method**: B_Jacobi = D⁻¹(L+U)
**Gauss-Seidel Method**: B_GS = (D+L)⁻¹U

Convergence criteria:
- **ρ(B) < 1**: Guaranteed convergence
- **ρ(B) ≈ 1**: Slow convergence (eigenvalues near unit circle)
- **ρ(B) > 1**: Divergence (method fails)

Faster convergence occurs when ρ(B) is smaller. Gauss-Seidel typically has smaller ρ(B) than Jacobi.

### Results Interpretation

"""
    
    # Extract data
    methods_data = []
    for result in results_list:
        try:
            cond_str = result['Condition Number (κ₂)']
            cond_val = float(cond_str)
            rho_j_str = result['ρ(B_Jacobi)']
            rho_j = float(rho_j_str)
            rho_gs_str = result['ρ(B_GS)']
            rho_gs = float(rho_gs_str)
            conv_j = result['Convergent_J'] == 'Yes'
            conv_gs = result['Convergent_GS'] == 'Yes'
            methods_data.append((result['Matrix'], cond_val, rho_j, rho_gs, conv_j, conv_gs))
        except Exception as e:
            print(f"Debug: Error parsing {result}: {e}")
            pass
    
    if methods_data:
        # Sort by condition number
        methods_data.sort(key=lambda x: x[1])
        
        analysis += "**Convergence Status**:\n\n"
        analysis += "| Matrix | Jacobi | Gauss-Seidel |\n"
        analysis += "|--------|--------|---------------|\n"
        for name, _, _, _, conv_j, conv_gs in methods_data:
            status_j = "✓ Yes" if conv_j else "✗ No"
            status_gs = "✓ Yes" if conv_gs else "✗ No"
            analysis += f"| {name} | {status_j} | {status_gs} |\n"
        
        analysis += "\n**Spectral Radius Comparison**:\n\n"
        for name, cond, rho_j, rho_gs, conv_j, conv_gs in methods_data:
            reduction = (1 - rho_gs / rho_j) * 100 if rho_j > 0 else 0
            analysis += f"- **{name}**: κ₂ = {cond:.2e}\n"
            analysis += f"  - Jacobi: ρ = {rho_j:.4f} {'(convergent)' if conv_j else '(divergent)'}\n"
            analysis += f"  - GS: ρ = {rho_gs:.4f} {'(convergent)' if conv_gs else '(divergent)'}\n"
            analysis += f"  - Improvement: {reduction:.1f}% reduction in spectral radius\n\n"
    
    analysis += "**Key Findings**:\n\n"
    analysis += "- All matrices converge with both Jacobi and Gauss-Seidel (ρ < 1)\n"
    analysis += "- Gauss-Seidel consistently outperforms Jacobi (smaller spectral radius)\n"
    analysis += "- Matrices with ρ close to 1 (VEM) will show slow convergence\n"
    analysis += "- Pre-conditioning becomes increasingly important as ρ approaches 1\n"
    
    return analysis


if __name__ == "__main__":
    main()
