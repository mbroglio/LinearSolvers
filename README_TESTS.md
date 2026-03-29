# Test Suite - Linear Solvers

## Descrizione
Script Python per eseguire test completi sui solutori iterativi implementati, secondo le specifiche del progetto.

## File Principale
- **`run_tests.py`**: Script principale per eseguire tutti i test

## Esecuzione

### Esecuzione Standard
```bash
python run_tests.py
```

Questo comando:
1. Carica tutte le matrici dalla directory `data/` (spa1.mtx, spa2.mtx, vem1.mtx, vem2.mtx)
2. Testa tutti i 4 metodi (Jacobi, Gauss-Seidel, Gradient, Conjugate Gradient)
3. Usa le 4 tolleranze richieste: 10^-4, 10^-6, 10^-8, 10^-10
4. Genera tabelle e grafici nella directory `results/`

## Output Generato

### Directory `results/`
Dopo l'esecuzione, vengono creati i seguenti file:

#### Tabelle Riassuntive
- `summary_table.xlsx` / `summary_table.csv`: Tabella pivot con tutti i risultati
- `detailed_report_[matrice].xlsx/.csv`: Tabelle dettagliate per ogni matrice

#### Grafici (formato PNG ad alta risoluzione 300 DPI)

**Per ogni matrice:**
- `iterations_vs_tolerance_[matrice].png`: Numero di iterazioni vs tolleranza
- `time_vs_tolerance_[matrice].png`: Tempo di esecuzione vs tolleranza  
- `error_vs_tolerance_[matrice].png`: Errore relativo vs tolleranza

**Grafico comparativo:**
- `comparative_all_matrices.png`: Confronto di tutte le matrici in un unico grafico

## Risultati dei Test

### Panoramica
Tutti i test sono stati completati con successo:
- **4 matrici** testate
- **4 metodi** implementati
- **4 tolleranze** per ogni metodo
- **64 test totali** - tutti convergenti ✓

### Statistiche Principali

| Matrice | Dimensione | Densità | Metodo Più Veloce |
|---------|------------|---------|-------------------|
| spa1.mtx | 1000×1000 | 18.23% | Gauss-Seidel |
| spa2.mtx | 3000×3000 | 18.13% | Gauss-Seidel |
| vem1.mtx | 1681×1681 | 0.47% | Conjugate Gradient |
| vem2.mtx | 2601×2601 | 0.31% | Conjugate Gradient |

### Osservazioni
- **Conjugate Gradient** è il metodo più efficiente per matrici sparse (vem1, vem2)
- **Gauss-Seidel** è molto veloce per matrici più dense (spa1, spa2)
- **Gradient** richiede molte più iterazioni rispetto a Conjugate Gradient
- Tutti i metodi convergono anche con tolleranza 10^-10

## Validazione
I test seguono la procedura di validazione della specifica:
1. ✓ Soluzione esatta: x = [1, 1, ..., 1]
2. ✓ Termine noto: b = A @ x
3. ✓ Risoluzione con tutti i 4 metodi
4. ✓ Calcolo di errore relativo, iterazioni e tempo
5. ✓ Test con tolleranze: 10^-4, 10^-6, 10^-8, 10^-10

## Script Rapido per Grafici

### `plot_results.py` - Genera grafici senza rieseguire i test

Dopo aver eseguito `run_tests.py` una volta, usa questo script per rigenerare grafici velocemente:

```bash
# Genera tutti i grafici (esecuzione rapida, ~5 secondi)
python plot_results.py

# Genera solo grafici specifici
python plot_results.py --plots iterations time

# Mostra grafici interattivamente invece di salvarli
python plot_results.py --show

# Salva in una directory diversa
python plot_results.py --output custom_results/

# Cambia risoluzione
python plot_results.py --dpi 150
```

**Opzioni disponibili:**
- `--plots`: `iterations`, `time`, `error`, `comparative`, `comparison`, `all` (default)
- `--output`: Directory di output (default: `results`)
- `--show`: Mostra grafici invece di salvarli
- `--dpi`: Risoluzione grafici (default: 300)

**Vantaggi:**
- ⚡ **Velocissimo**: ~5 secondi invece di 15-20 minuti
- 🎨 Rigenera grafici con stili diversi senza rifare i test
- 📊 Include nuovo grafico "method_comparison" con confronto a 3 pannelli

## Requisiti
```bash
pip install numpy pandas matplotlib openpyxl scipy
```

## Note
- I test (`run_tests.py`) richiedono circa 15-20 minuti per completare
- I grafici (`plot_results.py`) richiedono circa 5 secondi
- Utilizzare lo stesso hardware per confronti coerenti
- Il criterio di arresto è: ||Ax^(k) - b|| / ||b|| < tol
- Numero massimo di iterazioni: 20000
