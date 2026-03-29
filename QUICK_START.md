# Quick Start Guide - Plotting dei Risultati

## Script Disponibili

### 1. `run_tests.py` - Esecuzione Test Completa (~15-20 minuti)
Esegue tutti i test sui 4 metodi iterativi con 4 tolleranze diverse.

```bash
python run_tests.py
```

**Quando usarlo:** Solo la prima volta o quando si modificano i solutori.

### 2. `plot_results.py` - Generazione Rapida Grafici (~5-10 secondi)
Carica i dati già calcolati e genera grafici personalizzati.

```bash
# Genera tutti i grafici (default)
python plot_results.py

# Genera solo specifici tipi
python plot_results.py --plots iterations time
python plot_results.py --plots comparison
python plot_results.py --plots comparative error

# Cambia risoluzione
python plot_results.py --dpi 150      # Bassa risoluzione
python plot_results.py --dpi 600      # Alta risoluzione

# Output in directory diversa
python plot_results.py --output my_plots/

# Mostra grafici interattivamente (non salva)
python plot_results.py --show
```

## Tipi di Grafici Disponibili

### `iterations` - Iterazioni vs Tolleranza
- 4 grafici (uno per matrice)
- Confronta tutti i metodi su scala logaritmica
- File: `iterations_vs_tolerance_[matrice].png`

### `time` - Tempo di Esecuzione vs Tolleranza  
- 4 grafici (uno per matrice)
- Mostra efficienza computazionale
- File: `time_vs_tolerance_[matrice].png`

### `error` - Errore Relativo vs Tolleranza
- 4 grafici (uno per matrice)
- Scala log-log per visualizzare convergenza
- File: `error_vs_tolerance_[matrice].png`

### `comparative` - Confronto Tutte le Matrici
- 1 grafico con 4 subplot
- Vista panoramica di tutti i risultati
- File: `comparative_all_matrices.png`

### `comparison` - Confronto Dettagliato Metodi (NUOVO!)
- 4 grafici a 3 pannelli (uno per matrice)
- Mostra iterazioni, tempo ed errore affiancati
- Ideale per la relazione
- File: `method_comparison_[matrice].png`

## Esempi di Utilizzo Comuni

```bash
# Rigenera tutto velocemente
python plot_results.py

# Solo grafici per la relazione
python plot_results.py --plots comparison comparative

# Esperimenti con diverse risoluzioni
python plot_results.py --plots comparison --dpi 150 --output low_res/
python plot_results.py --plots comparison --dpi 600 --output high_res/

# Anteprima interattiva prima di salvare
python plot_results.py --plots comparison --show

# Genera solo grafici di performance
python plot_results.py --plots time iterations
```

## Confronto Tempi di Esecuzione

| Operazione | Tempo | Quando usare |
|------------|-------|--------------|
| `run_tests.py` | ~15-20 min | Prima esecuzione o modifiche ai solutori |
| `plot_results.py` | ~5-10 sec | Ogni volta che vuoi rigenerare/personalizzare grafici |
| `plot_results.py --plots comparison` | ~3-5 sec | Solo grafici specifici |

## Workflow Consigliato

1. **Prima volta:**
   ```bash
   python run_tests.py
   ```
   Questo genera i dati in `results/detailed_report_*.csv`

2. **Successive volte:**
   ```bash
   python plot_results.py
   ```
   Rigenera rapidamente tutti i grafici dai dati salvati

3. **Sperimentazione:**
   ```bash
   python plot_results.py --plots comparison --dpi 600
   python plot_results.py --plots error --show
   ```
   Prova diverse configurazioni e visualizzazioni

## Dati Salvati

I risultati dei test sono salvati in:
- `results/detailed_report_spa1.csv`
- `results/detailed_report_spa2.csv`
- `results/detailed_report_vem1.csv`
- `results/detailed_report_vem2.csv`
- `results/summary_table.xlsx` (tabella pivot)

Puoi anche modificare/analizzare questi CSV direttamente!

## Note

- I grafici sono salvati in formato PNG ad alta risoluzione (300 DPI default)
- Tutti i grafici usano scale logaritmiche dove appropriato
- Le statistiche riassuntive sono sempre stampate alla fine
- I dati vengono caricati automaticamente dai CSV in `results/`
