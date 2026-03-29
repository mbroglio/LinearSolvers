"""
Script veloce per generare grafici dai risultati già calcolati.
Carica i dati da CSV e genera grafici personalizzabili senza rieseguire i test.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results():
    """Carica i risultati dai file CSV."""
    results_dir = Path("results")
    
    # Carica tutti i detailed reports
    all_data = []
    for csv_file in results_dir.glob("detailed_report_*.csv"):
        df = pd.read_csv(csv_file)
        
        # Estrai il nome della matrice dal nome del file
        # es: detailed_report_spa1.csv -> spa1.mtx
        matrix_name = csv_file.stem.replace("detailed_report_", "") + ".mtx"
        df['matrix'] = matrix_name
        
        all_data.append(df)
    
    if not all_data:
        print("ERRORE: Nessun file di risultati trovato in results/")
        print("Eseguire prima: python run_tests.py")
        return None
    
    # Combina tutti i dati
    df = pd.concat(all_data, ignore_index=True)
    print(f"OK: Caricati {len(df)} risultati da {len(all_data)} file")
    return df


def plot_iterations_vs_tolerance(df, output_dir="results", show=False):
    """Genera grafici iterazioni vs tolleranza per ogni matrice."""
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
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
        
        output_path = Path(output_dir) / f"iterations_vs_tolerance_{matrix.replace('.mtx', '')}.png"
        plt.savefig(output_path, dpi=300)
        print(f"OK: Salvato: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_time_vs_tolerance(df, output_dir="results", show=False):
    """Genera grafici tempo vs tolleranza per ogni matrice."""
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
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
        
        output_path = Path(output_dir) / f"time_vs_tolerance_{matrix.replace('.mtx', '')}.png"
        plt.savefig(output_path, dpi=300)
        print(f"OK: Salvato: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_error_vs_tolerance(df, output_dir="results", show=False):
    """Genera grafici errore relativo vs tolleranza per ogni matrice."""
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        fig, ax = plt.subplots(figsize=(12, 7))
        matrix_data = df[df['matrix'] == matrix]
        
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            
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
        
        output_path = Path(output_dir) / f"error_vs_tolerance_{matrix.replace('.mtx', '')}.png"
        plt.savefig(output_path, dpi=300)
        print(f"OK: Salvato: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_comparative(df, output_dir="results", show=False):
    """Genera grafico comparativo con tutte le matrici."""
    matrices = df['matrix'].unique()
    
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
    
    output_path = Path(output_dir) / "comparative_all_matrices.png"
    plt.savefig(output_path, dpi=300)
    print(f"OK: Salvato: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_method_comparison(df, output_dir="results", show=False):
    """Confronto diretto tra metodi per ogni matrice."""
    matrices = df['matrix'].unique()
    
    for matrix in matrices:
        matrix_data = df[df['matrix'] == matrix]
        
        # Subplot con 3 grafici
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Iterazioni
        ax = axes[0]
        methods = matrix_data['method'].unique()
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            converged = method_data[method_data['converged'] == True]
            if len(converged) > 0:
                ax.plot(converged['tolerance'], converged['iterations'], 
                       marker='o', label=method, linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance', fontsize=11)
        ax.set_ylabel('Iterations', fontsize=11)
        ax.set_title('Iterations', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Tempo
        ax = axes[1]
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            converged = method_data[method_data['converged'] == True]
            if len(converged) > 0:
                ax.plot(converged['tolerance'], converged['time'], 
                       marker='s', label=method, linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance', fontsize=11)
        ax.set_ylabel('Time (s)', fontsize=11)
        ax.set_title('Execution Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Errore
        ax = axes[2]
        for method in methods:
            method_data = matrix_data[matrix_data['method'] == method]
            method_data = method_data.sort_values('tolerance')
            converged = method_data[
                (method_data['converged'] == True) & 
                (~method_data['relative_error'].isna())
            ]
            if len(converged) > 0:
                ax.plot(converged['tolerance'], converged['relative_error'], 
                       marker='^', label=method, linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Tolerance', fontsize=11)
        ax.set_ylabel('Relative Error', fontsize=11)
        ax.set_title('Relative Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Method Comparison - {matrix}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = Path(output_dir) / f"method_comparison_{matrix.replace('.mtx', '')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"OK: Salvato: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def print_summary(df):
    """Stampa statistiche riassuntive."""
    print("\n" + "="*80)
    print("STATISTICHE RIASSUNTIVE")
    print("="*80)
    
    for matrix in df['matrix'].unique():
        matrix_data = df[df['matrix'] == matrix]
        print(f"\n{matrix}:")
        
        for method in matrix_data['method'].unique():
            method_data = matrix_data[matrix_data['method'] == method]
            
            # Trova il risultato migliore (tolleranza più stretta)
            best = method_data.loc[method_data['tolerance'].idxmin()]
            
            print(f"  {method:25s}: {best['iterations']:5.0f} iter, "
                  f"{best['time']:8.4f}s @ tol={best['tolerance']:.0e}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera grafici dai risultati dei test già calcolati"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["iterations", "time", "error", "comparative", "comparison", "all"],
        default=["all"],
        help="Tipi di grafici da generare (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory di output (default: results)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra i grafici invece di salvarli solamente"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI per i grafici salvati (default: 300)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATORE GRAFICI VELOCE")
    print("="*80)
    
    # Carica i dati
    df = load_results()
    if df is None:
        return
    
    # Crea directory di output
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Aggiorna DPI globale
    plt.rcParams['figure.dpi'] = args.dpi
    plt.rcParams['savefig.dpi'] = args.dpi
    
    print(f"\nGenerazione grafici...")
    print(f"Output: {output_dir}/")
    
    # Determina quali grafici generare
    plot_types = args.plots
    if "all" in plot_types:
        plot_types = ["iterations", "time", "error", "comparative", "comparison"]
    
    # Genera i grafici richiesti
    if "iterations" in plot_types:
        print("\nGrafici iterazioni vs tolleranza...")
        plot_iterations_vs_tolerance(df, args.output, args.show)
    
    if "time" in plot_types:
        print("\nGrafici tempo vs tolleranza...")
        plot_time_vs_tolerance(df, args.output, args.show)
    
    if "error" in plot_types:
        print("\nGrafici errore vs tolleranza...")
        plot_error_vs_tolerance(df, args.output, args.show)
    
    if "comparative" in plot_types:
        print("\nGrafico comparativo...")
        plot_comparative(df, args.output, args.show)
    
    if "comparison" in plot_types:
        print("\nGrafici confronto metodi...")
        plot_method_comparison(df, args.output, args.show)
    
    # Stampa statistiche
    print_summary(df)
    
    print("\n" + "="*80)
    print(f"FATTO! Grafici generati in: {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
