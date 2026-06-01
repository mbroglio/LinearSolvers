from time import perf_counter
import numpy as np
from .base import IterativeSolver, SolverResult


class JacobiSolver(IterativeSolver):

    @property
    def name(self):
        return "Jacobi"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = A.diagonal()  # estrae il vettore della diagonale di A
        if np.any(
            np.isclose(diag, 0.0)
        ):  # controlla se almeno un elemento diagonale è vicino a zero
            raise ValueError("Jacobi requires non-zero diagonal")

        n = b.shape[0]  # numero di righe/incognite
        x = np.zeros(n)  # vettore soluzione inizializzato a zero

        norm_b = np.linalg.norm(
            b
        )  # norma euclidea di b (usata per normalizzare il residuo)
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        A_offdiag = A.copy()  # crea una copia di A
        A_offdiag.setdiag(0)  # azzera la diagonale di A_offdiag

        for k in range(self.max_iter):
            Ax_b = A @ x - b  # prodotto matrice-vettore e calcolo del residuo
            rel_res = np.linalg.norm(Ax_b) / norm_b  # residuo relativo
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            # Optimization: Write result into existing array x instead of creating a new one
            x[:] = (
                b - A_offdiag @ x
            ) / diag  # aggiornamento Jacobi: mat-vec con soli elementi off-diagonale, divisione per la diagonale

        rel_res = (
            np.linalg.norm(A @ x - b) / norm_b
        )  # residuo relativo finale dopo max_iter iterazioni
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class GaussSeidelSolver(IterativeSolver):

    @property
    def name(self):
        return "Gauss-Seidel"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = A.diagonal()  # estrae il vettore della diagonale di A
        if np.any(
            np.isclose(diag, 0.0)
        ):  # controlla se almeno un elemento diagonale è vicino a zero
            raise ValueError("Gauss-Seidel requires non-zero diagonal")

        n = A.shape[0]  # dimensione del sistema
        x = np.zeros(n)  # vettore soluzione inizializzato a zero

        norm_b = np.linalg.norm(b)  # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        for k in range(self.max_iter):
            Ax_b = A @ x - b  # prodotto matrice-vettore e calcolo del residuo
            rel_res = np.linalg.norm(Ax_b) / norm_b  # residuo relativo
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            for i in range(n):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                data = A.data[row_start:row_end]
                indices = A.indices[row_start:row_end]

                # Optimization: Vectorized dot product mapped exactly to memory indices
                sum_Ax = (
                    np.dot(data, x[indices]) - diag[i] * x[i]
                )  # prodotto scalare riga i escluso il termine diagonale
                x[i] = (b[i] - sum_Ax) / diag[
                    i
                ]  # aggiornamento Gauss-Seidel per la componente i

        rel_res = (
            np.linalg.norm(A @ x - b) / norm_b
        )  # residuo relativo finale dopo max_iter iterazioni
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class GradientSolver(IterativeSolver):

    @property
    def name(self):
        return "Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if (
            np.abs(A - A.T).max() > 1e-10
        ):  # verifica simmetria: massimo scarto tra A e la sua trasposta
            raise ValueError("Gradient requires symmetric matrix")

        n = b.shape[0]  # dimensione del sistema
        x = np.zeros(n)  # vettore soluzione inizializzato a zero
        r = b.copy()  # residuo iniziale r = b (perché x = 0)

        norm_b = np.linalg.norm(b)  # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        for k in range(self.max_iter):
            # Optimization: r is already the residual (b - Ax), no need to recompute A @ x
            rel_res = (
                np.linalg.norm(r) / norm_b
            )  # norma del residuo corrente divisa per ‖b‖
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            Ar = A @ r  # prodotto matrice-vettore A*r
            denom = np.dot(r, Ar)  # prodotto scalare r^T A r (deve essere > 0 per SPD)
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")

            alpha = np.dot(r, r) / denom  # passo ottimale: ‖r‖² / (r^T A r)

            # Optimization: In-place arithmetic avoids array reallocation
            x += alpha * r
            r -= alpha * Ar

        rel_res = self._relative_residual(A, x, b)
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)


class ConjugateGradientSolver(IterativeSolver):

    @property
    def name(self):
        return "Conjugate Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if (
            np.abs(A - A.T).max() > 1e-10
        ):  # verifica simmetria: massimo scarto tra A e la sua trasposta
            raise ValueError("Conjugate Gradient requires symmetric matrix")

        n = b.shape[0]  # dimensione del sistema
        x = np.zeros(n)  # vettore soluzione inizializzato a zero
        r = b.copy()  # residuo iniziale r = b (perché x = 0)
        p = r.copy()  # direzione di discesa iniziale p = r

        norm_b = np.linalg.norm(b)  # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0

        start = perf_counter()

        for k in range(self.max_iter):
            # Optimization: r is already the residual (b - Ax), no need to recompute A @ x
            rel_res = (
                np.linalg.norm(r) / norm_b
            )  # norma del residuo corrente divisa per ‖b‖
            if rel_res < self.tol:
                return SolverResult(x, k, perf_counter() - start, rel_res, True)

            Ap = A @ p  # prodotto matrice-vettore A*p
            denom = np.dot(p, Ap)  # prodotto scalare p^T A p
            if denom <= 0.0:
                raise ValueError("Matrix not positive definite")

            rr_old = np.dot(r, r)  # ‖r‖² prima dell'aggiornamento
            alpha = rr_old / denom  # passo ottimale lungo la direzione p

            # Optimization: In-place arithmetic avoids array reallocation
            x += alpha * p
            r -= alpha * Ap

            rr_new = np.dot(r, r)  # ‖r‖² dopo l'aggiornamento
            beta = (
                rr_new / rr_old
            )  # coefficiente di correzione della direzione coniugata

            # Optimization: Update p in-place
            p *= beta
            p += r

        rel_res = self._relative_residual(
            A, x, b
        )  # residuo relativo finale: ‖Ax - b‖ / ‖b‖
        return SolverResult(x, self.max_iter, perf_counter() - start, rel_res, False)
