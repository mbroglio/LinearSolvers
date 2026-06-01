from time import perf_counter
import numpy as np
from numba import njit
from .base import IterativeSolver, SolverResult


@njit(fastmath=True)
def _csr_matvec(data, indices, indptr, n, x):
    res = np.zeros(n)                              # vettore risultato inizializzato a zero
    for i in range(n):
        dot = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            dot += data[p] * x[indices[p]]
        res[i] = dot
    return res


@njit(fastmath=True)
def _jacobi_core(data, indices, indptr, b, diag, n, max_iter, tol, norm_b):
    x = np.zeros(n)                               # vettore soluzione inizializzato a zero
    x_new = np.zeros(n)                           # vettore temporaneo per l'aggiornamento Jacobi
    for k in range(max_iter):
        norm_res = 0.0
        for i in range(n):
            Ax_i = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                Ax_i += data[p] * x[indices[p]]
            res_i = Ax_i - b[i]
            norm_res += res_i * res_i
            x_new[i] = x[i] - res_i / diag[i]

        rel_res = np.sqrt(norm_res) / norm_b      # radice quadrata della somma dei quadrati = norma euclidea del residuo
        if rel_res < tol:
            return x_new, k, rel_res, True

        for i in range(n):
            x[i] = x_new[i]

    norm_res = 0.0
    for i in range(n):
        Ax_i = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            Ax_i += data[p] * x[indices[p]]
        res_i = Ax_i - b[i]
        norm_res += res_i * res_i
    rel_res = np.sqrt(norm_res) / norm_b          # residuo relativo finale
    return x, max_iter, rel_res, False


class JacobiSolver(IterativeSolver):
    @property
    def name(self):
        return "Jacobi"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = A.diagonal()                       # estrae il vettore della diagonale di A
        if np.any(np.abs(diag) < 1e-15):          # controlla se almeno un elemento diagonale è (quasi) zero
            raise ValueError("Jacobi requires non-zero diagonal")
        norm_b = np.linalg.norm(b)                # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _jacobi_core(
            A.data,
            A.indices,
            A.indptr,
            b,
            diag,
            b.shape[0],
            self.max_iter,
            self.tol,
            norm_b,
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


@njit(fastmath=True)
def _gs_core(data, indices, indptr, b, diag, n, max_iter, tol, norm_b):
    x = np.zeros(n)                               # vettore soluzione inizializzato a zero
    for k in range(max_iter):
        norm_res = 0.0
        for i in range(n):
            Ax_i = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                Ax_i += data[p] * x[indices[p]]
            res_i = Ax_i - b[i]
            norm_res += res_i * res_i

        rel_res = np.sqrt(norm_res) / norm_b      # norma euclidea del residuo divisa per ‖b‖
        if rel_res < tol:
            return x, k, rel_res, True

        for i in range(n):
            sum_Ax = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if i != j:
                    sum_Ax += data[p] * x[j]
            x[i] = (b[i] - sum_Ax) / diag[i]

    norm_res = 0.0
    for i in range(n):
        Ax_i = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            Ax_i += data[p] * x[indices[p]]
        res_i = Ax_i - b[i]
        norm_res += res_i * res_i
    rel_res = np.sqrt(norm_res) / norm_b          # residuo relativo finale
    return x, max_iter, rel_res, False


class GaussSeidelSolver(IterativeSolver):
    @property
    def name(self):
        return "Gauss-Seidel"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        diag = A.diagonal()                       # estrae il vettore della diagonale di A
        if np.any(np.abs(diag) < 1e-15):          # controlla se almeno un elemento diagonale è (quasi) zero
            raise ValueError("Gauss-Seidel requires non-zero diagonal")
        norm_b = np.linalg.norm(b)                # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _gs_core(
            A.data,
            A.indices,
            A.indptr,
            b,
            diag,
            A.shape[0],
            self.max_iter,
            self.tol,
            norm_b,
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


@njit(fastmath=True)
def _grad_core(data, indices, indptr, b, n, max_iter, tol, norm_b):
    x = np.zeros(n)                               # vettore soluzione inizializzato a zero
    r = b.copy()                                  # residuo iniziale r = b (perché x = 0)
    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b      # norma del residuo corrente divisa per ‖b‖
        if rel_res < tol:
            return x, k, rel_res, True
        Ar = _csr_matvec(data, indices, indptr, n, r)
        denom = np.dot(r, Ar)                     # prodotto scalare r^T A r
        if denom <= 0.0:
            return x, k, rel_res, False
        alpha = np.dot(r, r) / denom              # passo ottimale: ‖r‖² / (r^T A r)
        x = x + alpha * r
        r = r - alpha * Ar
    rel_res = np.linalg.norm(r) / norm_b          # residuo relativo finale
    return x, max_iter, rel_res, False


class GradientSolver(IterativeSolver):
    @property
    def name(self):
        return "Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if np.abs(A - A.T).max() > 1e-10:         # verifica simmetria: massimo scarto tra A e la sua trasposta
            raise ValueError("Gradient requires symmetric matrix")
        norm_b = np.linalg.norm(b)                # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _grad_core(
            A.data, A.indices, A.indptr, b, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)


@njit(fastmath=True)
def _cg_core(data, indices, indptr, b, n, max_iter, tol, norm_b):
    x = np.zeros(n)                               # vettore soluzione inizializzato a zero
    r = b.copy()                                  # residuo iniziale r = b (perché x = 0)
    p = r.copy()                                  # direzione di discesa iniziale p = r
    for k in range(max_iter):
        rel_res = np.linalg.norm(r) / norm_b      # norma del residuo corrente divisa per ‖b‖
        if rel_res < tol:
            return x, k, rel_res, True
        Ap = _csr_matvec(data, indices, indptr, n, p)
        denom = np.dot(p, Ap)                     # prodotto scalare p^T A p
        if denom <= 0.0:
            return x, k, rel_res, False
        rr_old = np.dot(r, r)                     # ‖r‖² prima dell'aggiornamento
        alpha = rr_old / denom                    # passo ottimale lungo la direzione p
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = np.dot(r, r)                     # ‖r‖² dopo l'aggiornamento
        beta = rr_new / rr_old                    # coefficiente di correzione della direzione coniugata
        p = r + beta * p
    rel_res = np.linalg.norm(r) / norm_b          # residuo relativo finale
    return x, max_iter, rel_res, False


class ConjugateGradientSolver(IterativeSolver):
    @property
    def name(self):
        return "Conjugate Gradient"

    def solve(self, A, b):
        self._validate_inputs(A, b)
        if np.abs(A - A.T).max() > 1e-10:         # verifica simmetria: massimo scarto tra A e la sua trasposta
            raise ValueError("Conjugate Gradient requires symmetric matrix")
        norm_b = np.linalg.norm(b)                # norma euclidea di b
        if norm_b == 0:
            norm_b = 1.0
        start = perf_counter()
        x, k, rel_res, conv = _cg_core(
            A.data, A.indices, A.indptr, b, b.shape[0], self.max_iter, self.tol, norm_b
        )
        return SolverResult(x, k, perf_counter() - start, rel_res, conv)
