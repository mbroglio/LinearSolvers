"""
Script per eseguire test completi su tutte le matrici con diverse tolleranze.
Genera tabelle e grafici secondo le specifiche del progetto.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import pandas as pd
from src.linear_solvers import (
    JacobiSolver,
    GaussSeidelSolver,
    GradientSolver,
    ConjugateGradientSolver,
    load_matrix_market,
)


def run_complete_tests():
    """Esegue test completi secondo le specifiche del progetto."""
    
    # Matrici di test
    matrix_files = [
        "spa1.mtx",
        "spa2.mtx",
        "vem1.mtx",
        "vem2.mtx",
    ]
    
    # Tolleranze da testare (dalla specifica)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    # Parametri
    max_iter = 20000
    data_dir = Path("data")
    
    # Risultati per ogni matrice
    all_results = {}
    
    print("="*100)
    print("TEST SUITE - Linear Solvers Validation")
    print("="*100)
    print(f"Max iterations: {max_iter}")
    print(f"Tolerances: {tolerances}")
    print("="*100)
    print()
    
    # Itera su ogni matrice
    for matrix_file in matrix_files:
        matrix_path = data_dir / matrix_file
        
        if not matrix_path.exists():
            print(f"WARNING: Matrix {matrix_file} not found, skipping...")
            continue
            
        print(f"\n{'='*100}")
        print(f"Testing matrix: {matrix_file}")
        print(f"{'='*100}")
        
        # Carica matrice
        A = load_matrix_market(matrix_path)
        print(f"Matrix shape: {A.shape}")
        print(f"Matrix density: {np.count_nonzero(A) / A.size * 100:.2f}%")
        
        # Crea soluzione esatta e termine noto (Step 1 e 2 della specifica)
        x_exact = np.ones(A.shape[0])
        b = A @ x_exact
        
        # Risultati per questa matrice
        matrix_results = []
        
        # Itera su ogni tolleranza
        for tol in tolerances:
            print(f"\n{'-'*100}")
            print(f"Tolerance: {tol:.0e}")
            print(f"{'-'*100}")
            print(f"{'Method':<25} {'Conv':<8} {'RelErr':<14} {'RelRes':<14} {'Iters':<10} {'Time[s]':<12}")
            print(f"{'-'*100}")
            
            # Definisci i solutori
            solvers = [
                JacobiSolver(tol, max_iter),
                GaussSeidelSolver(tol, max_iter),
                GradientSolver(tol, max_iter),
                ConjugateGradientSolver(tol, max_iter),
            ]
            
            # Testa ogni solutore
            for solver in solvers:
                try:
                    # Risolvi il sistema (Step 3)
                    result = solver.solve(A, b)
                    
                    # Calcola errore relativo (Step 4)
                    rel_error = np.linalg.norm(result.solution - x_exact) / np.linalg.norm(x_exact)
                    
                    # Convergenza
                    conv_str = "Yes" if result.converged else "No"
                    
                    # Stampa risultati
                    print(
                        f"{solver.name:<25} {conv_str:<8} {rel_error:<14.6e} "
                        f"{result.relative_residual:<14.6e} {result.iterations:<10} "
                        f"{result.elapsed_seconds:<12.6f}"
                    )
                    
                    # Salva risultati
                    matrix_results.append({
                        'matrix': matrix_file,
                        'method': solver.name,
                        'tolerance': tol,
                        'converged': result.converged,
                        'relative_error': rel_error,
                        'relative_residual': result.relative_residual,
                        'iterations': result.iterations,
                        'time': result.elapsed_seconds,
                    })
                    
                except Exception as e:
                    print(f"{solver.name:<25} No       ERROR: {str(e)}")
                    
                    # Salva risultato di errore
                    matrix_results.append({
                        'matrix': matrix_file,
                        'method': solver.name,
                        'tolerance': tol,
                        'converged': False,
                        'relative_error': np.nan,
                        'relative_residual': np.nan,
                        'iterations': 0,
                        'time': 0.0,
                    })
        
        all_results[matrix_file] = matrix_results
    
    # Converti in DataFrame per analisi
    df = pd.DataFrame([item for results in all_results.values() for item in results])
    
    # Genera tabelle e grafici
    generate_tables_and_plots(df)
    
    print("\n" + "="*100)
    print("Test completati! Tabelle e grafici salvati nella directory corrente.")
    print("="*100)
    
    return df


def generate_tables_and_plots(df: pd.DataFrame):
    """Genera tabelle e grafici secondo le specifiche."""
    
    # Crea directory per output
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*100)
    print("Generazione tabelle e grafici...")
    print("="*100)
    
    # 1. Tabella riassuntiva per ogni matrice e tolleranza
    print("\nGenerazione tabella riassuntiva...")
    summary_table = df.pivot_table(
        index=['matrix', 'tolerance'],
        columns='method',
        values=['iterations', 'time', 'converged'],
        aggfunc='first'
    )
    summary_table.to_csv(output_dir / "summary_table.csv")
    summary_table.to_excel(output_dir / "summary_table.xlsx")
    print(f"✓ Tabella salvata in {output_dir / 'summary_table.xlsx'}")
    
    # 2. Grafici: Iterazioni vs Tolleranza per ogni matrice
    print("\nGenerazione grafici iterazioni vs tolleranza...")
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
            # Filtra solo convergenze
            converged_data = method_data[method_data['converged'] == True]
            
            if len(converged_data) > 0:
                ax.plot(
                    converged_data['tolerance'],
                    converged_data['iterations'],
                    marker='o',
                    label=method,
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance', fontsize=12)
        ax.set_ylabel('Iterations', fontsize=12)
        ax.set_title(f'Iterations vs Tolerance - {matrix}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"iterations_vs_tolerance_{matrix.replace('.mtx', '')}.png", dpi=300)
        plt.close()
    
    print(f"✓ Grafici iterazioni salvati in {output_dir}")
    
    # 2b. Grafici: Iterazioni vs Tolleranza per spa1 e spa2 (senza gradient)
    print("\nGenerazione grafici iterazioni vs tolleranza (spa1 e spa2, no gradient)...")
    spa_matrices = [m for m in matrices if 'spa1' in m or 'spa2' in m]
    
    for matrix in spa_matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            # Skip only gradient method (not conjugate gradient)
            if method.lower() == 'gradient':
                continue
                
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
            # Filtra solo convergenze
            converged_data = method_data[method_data['converged'] == True]
            
            if len(converged_data) > 0:
                ax.plot(
                    converged_data['tolerance'],
                    converged_data['iterations'],
                    marker='o',
                    label=method,
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance', fontsize=12)
        ax.set_ylabel('Iterations', fontsize=12)
        ax.set_title(f'Iterations vs Tolerance - {matrix} (no gradient)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"iterations_vs_tolerance_{matrix.replace('.mtx', '')}_no_gradient.png", dpi=300)
        plt.close()
    
    print(f"✓ Grafici spa1/spa2 (no gradient) salvati in {output_dir}")
    
    # 3. Grafici: Tempo vs Tolleranza per ogni matrice
    print("\nGenerazione grafici tempo vs tolleranza...")
    
    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
            # Filtra solo convergenze
            converged_data = method_data[method_data['converged'] == True]
            
            if len(converged_data) > 0:
                ax.plot(
                    converged_data['tolerance'],
                    converged_data['time'],
                    marker='s',
                    label=method,
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title(f'Execution Time vs Tolerance - {matrix}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"time_vs_tolerance_{matrix.replace('.mtx', '')}.png", dpi=300)
        plt.close()
    
    print(f"✓ Grafici tempo salvati in {output_dir}")
    
    # 4. Grafici: Errore relativo vs Tolleranza
    print("\nGenerazione grafici errore relativo vs tolleranza...")
    
    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
            # Filtra solo convergenze e errori validi
            converged_data = method_data[
                (method_data['converged'] == True) & 
                (~method_data['relative_error'].isna())
            ]
            
            if len(converged_data) > 0:
                ax.plot(
                    converged_data['tolerance'],
                    converged_data['relative_error'],
                    marker='^',
                    label=method,
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Tolerance', fontsize=12)
        ax.set_ylabel('Relative Error', fontsize=12)
        ax.set_title(f'Relative Error vs Tolerance - {matrix}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"error_vs_tolerance_{matrix.replace('.mtx', '')}.png", dpi=300)
        plt.close()
    
    print(f"✓ Grafici errore salvati in {output_dir}")
    
    # 5. Grafico comparativo: tutti i metodi su tutte le matrici (iterazioni)
    print("\nGenerazione grafico comparativo generale...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, matrix in enumerate(matrices):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
            converged_data = method_data[method_data['converged'] == True]
            
            if len(converged_data) > 0:
                ax.plot(
                    converged_data['tolerance'],
                    converged_data['iterations'],
                    marker='o',
                    label=method,
                    linewidth=2
                )
        
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance', fontsize=10)
        ax.set_ylabel('Iterations', fontsize=10)
        ax.set_title(f'{matrix}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_all_matrices.png", dpi=300)
    plt.close()
    
    print(f"✓ Grafico comparativo salvato in {output_dir}")
    
    # 6. Tabella dettagliata per la relazione
    print("\nGenerazione tabella dettagliata per relazione...")
    
    for matrix in matrices:
        matrix_data = df[df['matrix'] == matrix].copy()
        matrix_data = matrix_data.sort_values(['tolerance', 'method'])
        
        # Formatta per la relazione
        report_table = matrix_data[[
            'method', 'tolerance', 'converged', 
            'relative_error', 'iterations', 'time'
        ]].copy()
        
        report_table.to_csv(
            output_dir / f"detailed_report_{matrix.replace('.mtx', '')}.csv",
            index=False
        )
        report_table.to_excel(
            output_dir / f"detailed_report_{matrix.replace('.mtx', '')}.xlsx",
            index=False
        )
    
    print(f"✓ Tabelle dettagliate salvate in {output_dir}")
    
    # 7. Stampa statistiche finali
    print("\n" + "="*100)
    print("STATISTICHE FINALI")
    print("="*100)
    
    for matrix in matrices:
        print(f"\n{matrix}:")
        matrix_data = df[df['matrix'] == matrix]
        
        for method in matrix_data['method'].unique():
            method_data = matrix_data[matrix_data['method'] == method]
            converged_count = method_data['converged'].sum()
            total_count = len(method_data)
            
            print(f"  {method:25s}: {converged_count}/{total_count} convergenze")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    # Installa pandas e matplotlib se necessario
    try:
        import pandas
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib", "openpyxl"])
        print("Packages installed successfully!")
    
    # Esegui i test
    df = run_complete_tests()
    
    print("\n✓ Tutti i test completati con successo!")
    print(f"✓ Risultati disponibili nella directory 'results/'")
