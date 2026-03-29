from pathlib import Path
import numpy as np


def load_matrix_market(file_path):
    path = Path(file_path)
    with path.open("r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("%")
        ]

    if not lines:
        raise ValueError("Empty file")

    n_rows, n_cols, n_entries = map(int, lines[0].split())
    matrix = np.zeros((n_rows, n_cols))

    with path.open("r") as f:
        header = f.readline().strip()
        is_symmetric = "symmetric" in header.lower()
        is_pattern = "pattern" in header.lower()

    for line in lines[1 : n_entries + 1]:
        tokens = line.split()
        i = int(tokens[0]) - 1
        j = int(tokens[1]) - 1
        value = 1.0 if is_pattern else float(tokens[2])
        matrix[i, j] = value
        if is_symmetric and i != j:
            matrix[j, i] = value

    return matrix
