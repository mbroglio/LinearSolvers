# Progetto 1: Algebra Lineare Numerica
**Metodi del Calcolo Scientifico - AA 2025-2026**

---

## 1. Descrizione Generale
L'obiettivo del progetto è l'implementazione di una **mini libreria** per la risoluzione di sistemi lineari, limitatamente al caso di **matrici simmetriche e definite positive**.

### Solutori richiesti:
1.  Metodo di Jacobi
2.  Metodo di Gauß-Seidel
3.  Metodo del Gradiente
4.  Metodo del Gradiente coniugato

### Requisiti tecnologici:
* **Linguaggio:** Qualsiasi linguaggio OPEN SOURCE (es. C++, Fortran, Java, Python).
* **Gestione Dati:** È richiesto l'utilizzo di librerie open-source (come Eigen, Armadillo, blas/lapack) o funzionalità native del linguaggio per le strutture dati e le operazioni elementari.
* **Vincolo Fondamentale:** La libreria di base deve fornire **solo** la struttura dati di matrici e vettori. **Non è consentito** l'uso di solutori già implementati internamente alle librerie scelte (es. il metodo di Jacobi di Eigen non è ammesso).

---

## 2. Requisiti della Libreria
* **Architettura:** Sarà valutata positivamente una struttura ben organizzata del codice rispetto a una semplice sequenza di funzioni indipendenti.
* **Criteri di Arresto:** * Il vettore iniziale deve essere nullo ($x^{(0)} = 0$).
    * La condizione di arresto basata sulla tolleranza ($tol$) assegnata dall'utente è:
        $$\frac{||Ax^{(k)}-b||}{||b||} < tol$$
    * Deve essere presente un controllo sul numero massimo di iterazioni ($maxIter \ge 20000$). Se superato, il codice deve segnalare la mancata convergenza.

---

## 3. Validazione e Test
Il codice deve essere un eseguibile che accetta in input una matrice $A$ (preferibilmente in formato `.mtx`) e una tolleranza $tol$.

### Procedura di validazione:
Per le matrici fornite (`spa1.mtx`, `spa2.mtx`, `vem1.mtx`, `vem2.mtx`), seguire questi step:
1.  **Step 1:** Creare un vettore soluzione esatta $x$ con tutte le entrate uguali a 1.
2.  **Step 2:** Calcolare il termine noto $b = Ax$.
3.  **Step 3:** Risolvere il sistema con i quattro metodi implementati.
4.  **Step 4:** Calcolare e visualizzare a schermo:
    * Errore relativo tra la soluzione esatta e quella computata.
    * Numero di iterazioni effettuate.
    * Tempo di calcolo.

**Nota sui test:** I test devono essere eseguiti per diverse tolleranze (10^-4, 10^-6, 10^-8 e 10^-10), partendo da $tol = 10^{-10}$, mantenendo lo stesso hardware per i confronti.

---

## 4. Note Tecniche e Formato .mtx
* Il formato **.mtx (Matrix Market)** è uno standard ASCII per lo scambio di matrici.
* La numerazione delle entrate nel file parte da **1**.
* Maggiori informazioni: [https://math.nist.gov/MatrixMarket/formats.html](https://math.nist.gov/MatrixMarket/formats.html).

---

## 5. Esame Finale
* **Modalità:** Portare il computer portatile con tutti i codici. Potrebbe essere richiesto di eseguire test estemporanei ("on-the-fly") su nuove matrici.
* **Relazione:** Preparare una breve relazione con:
    * Descrizione dell'architettura della libreria.
    * Risultati organizzati in tabelle o grafici.
    * Commenti e conclusioni.
* **Scadenza:** La relazione va inviata **72 ore (3 giorni) prima** dell'esame.

---
**Contatti:** In caso di dubbi, contattare il docente via email.