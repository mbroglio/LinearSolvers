import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os


def plot_sparsity(matrix_path, title):
    A = scipy.io.mmread(matrix_path)
    plt.figure(figsize=(6, 6))
    plt.spy(A, markersize=1)
    plt.title(f"Sparsity Pattern - {title}")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_spy.png")
    plt.close()


if __name__ == "__main__":
    matrices = {
        "spa1": "../data/spa1.mtx",
        "spa2": "../data/spa2.mtx",
        "vem1": "../data/vem1.mtx",
        "vem2": "../data/vem2.mtx",
    }

    for name, path in matrices.items():
        if os.path.exists(path):
            plot_sparsity(path, name)
            print(f"Generated plot for {name}")
        else:
            print(f"Matrix file not found: {path}")
