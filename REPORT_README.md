# LaTeX Report Generation Summary

## File Generated
**report.tex** - Complete LaTeX report for the Linear Solvers project

## Report Structure

The report has been generated according to the specification and includes:

### 1. Introduction
Brief overview of the project goals and implemented methods.

### 2. Library Structure (Struttura della Libreria)
- **File Organization**: Description of how the project is organized into folders:
  - `src/linear_solvers/`: main package with base classes, implementations, and I/O utilities
  - `data/`: test matrices in Matrix Market format
  - `results/`: output tables and graphs
  - `main.py`: validation script
  - `plot_results.py`: plotting script

- **OOP Architecture**: Explanation of the Template Method pattern with base class `IterativeSolver`

### 3. Implemented Methods (Metodi Iterativi Implementati)
For each of the four methods (Jacobi, Gauss-Seidel, Gradient, Conjugate Gradient):
- **Description**: Brief explanation of how the method works
- **Algorithm**: Step-by-step mathematical formulation including:
  - Input parameters
  - Initialization
  - Iteration computation
  - Stopping criteria
- **Implementation**: Key implementation details

### 4. Experimental Results (Risultati Sperimentali)
- **Experimental Setup**: Description of test matrices and validation procedure
- **Summary Table**: Comprehensive comparison table with:
  - Number of iterations
  - Relative error
  - Execution time
  - Convergence status
- **Analysis**: Detailed commentary on:
  - Convergence behavior
  - Iteration count comparison
  - Execution time analysis
  - Precision achieved
  - Tolerance effect

### 5. Conclusions (Conclusioni)
- Synthesis of results
- Recommendations for practical use
- Future developments

## Compiling the Report

To generate the PDF from the LaTeX source:

### Option 1: Using pdflatex (recommended)
```bash
cd "path/to/LinearSolvers"
pdflatex report.tex
pdflatex report.tex  # Run twice for table of contents and references
```

### Option 2: Using online LaTeX editor
1. Open https://www.overleaf.com/
2. Create a new blank project
3. Upload `report.tex`
4. Click "Recompile"

### Option 3: Using TeXworks, TeXstudio, or similar
1. Open `report.tex` in your LaTeX editor
2. Click the "Build" or "Compile" button
3. Select pdflatex as the engine

## Required LaTeX Packages
The report uses the following packages (usually included in standard LaTeX distributions):
- inputenc (UTF-8 encoding)
- babel (Italian language)
- geometry (page margins)
- amsmath, amssymb (mathematical symbols)
- graphicx (for future graphics inclusion)
- booktabs (professional tables)
- float (table positioning)
- hyperref (clickable references)
- listings (code formatting)
- xcolor (colors)

## Notes
- The report is written in **Italian** as per the specification
- All mathematical formulas are properly formatted
- The table uses actual data from `results/summary_table.csv`
- The structure follows exactly the specification requirements
