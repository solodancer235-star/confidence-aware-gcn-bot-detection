# Confidence-Aware GCN for Social Bot Detection

Code, experiments and paper for **"Confidence-Aware Graph Convolutional Networks
for Social Bot Detection: A Message-Passing Reweighting Approach"**
(K. Kassymkhan, Astana IT University).

A Confidence-Aware GCN reweights every message `j → i` by the model's own
confidence in neighbour `j` (max softmax probability from the previous epoch),
renormalised over the incoming edges of `i`:

```
c_j = max_k softmax(z_j)_k          w_{j→i} = c_j^α / Σ_{k∈N(i)} c_k^α
h_i' = σ( Σ_{j∈N(i)} w_{j→i} · W h_j )
```

No extra learnable parameters; confidence is a plain tensor updated under
`torch.no_grad()` after each epoch.

## Main results (Twitter, 5 000 accounts, 5-fold CV × 5 seeds)

| Model | Prec. | Rec. | F1 | AUROC |
|---|---|---|---|---|
| MLP (no graph) | 0.805 | 0.809 | **0.807** | **0.890** |
| GCN | 0.731 | 0.752 | 0.740 | 0.819 |
| GraphSAGE | 0.805 | 0.801 | 0.803 | 0.887 |
| GAT | 0.744 | 0.726 | 0.734 | 0.819 |
| Conf-GCN | 0.732 | 0.732 | 0.731 | 0.794 |

* The +3.0-point precision gain from the earlier single-split experiment **does
  not replicate** (Conf-GCN vs GCN precision 0.732 vs 0.731, p = 0.83).
* Uniform / shuffled / inverted confidence reproduce Conf-GCN → the effect comes
  from the normalisation (mean aggregation, no self-loop), not from confidence.
  Confidence weights deviate from uniform by only ~6.5 %.
* **Graph construction matters most:** a k-NN graph over description + metadata
  (homophily 0.76 vs 0.62) lifts every GNN above the MLP
  (GraphSAGE 0.825, GCN 0.820, Conf-GCN 0.818, GAT 0.814 F1).

Full tables: [`results/summary.md`](results/summary.md); paper:
[`paper/main.pdf`](paper/main.pdf) (source `paper/main.tex`).

## Repository layout

```
confgcn/            library
  layers.py         ConfidenceWeightedConv (PyG MessagePassing)
  models.py         MLP, GCN, GraphSAGE, GAT, ConfidenceAwareGCN (+ ablation modes)
  data.py           dataset loading, SBERT features, k-NN graphs, synthetic graph
  train.py          training loop, CV splits, metrics, Wilcoxon + Holm
scripts/
  prepare_data.py   5 000-account subsample + SBERT embeddings
  run_synthetic.py  synthetic graph (RQ1) + Fig. 1
  run_experiments.py  hyper-parameter selection, main comparison, ablations,
                      graph density, graph construction (resumable)
  analyze.py        all tables (paper/tables/*.tex) and figures
  diagnostics.py    why confidence has no effect (weight spread, informativeness)
tests/              unit tests of the layer's formal properties
results/            per-run CSVs, predicted probabilities, summaries
paper/              LaTeX source (IEEEtran), figures, compiled PDF
data/               dataset (see data/README.md)
legacy/             original diploma prototype (synthetic graph, Leiden, Reddit collector)
```

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh            # ~2 h on a 2-core CPU
```

or step by step:

```bash
python scripts/prepare_data.py                     # builds data/processed/twitter_5000.npz
python -m pytest -q tests
python scripts/run_synthetic.py
python scripts/run_experiments.py --group select   # hyper-parameter selection (val only)
python scripts/run_experiments.py --group main ablation density graph
python scripts/diagnostics.py
python scripts/analyze.py
cd paper && latexmk -pdf main.tex
```

Use the original Kaggle CSV instead of the cleaned parquet with
`python scripts/prepare_data.py --data twitter_human_bots_dataset.csv`.
If Hugging Face is unreachable, point `SBERT_MODEL` at a local copy of
`all-MiniLM-L6-v2`.

## Protocol

* stratified 5-fold CV × 5 seeds (25 runs/model), 60/20/20 train/val/test;
* lr ∈ {0.01, 0.005}, weight decay ∈ {5e-4, 5e-5}, GAT heads ∈ {1, 4, 8},
  selected by validation F1 on seed 0; 100 epochs, no early stopping;
* metrics: accuracy, precision, recall, F1, AUROC, AUC-PR, ECE;
* paired Wilcoxon signed-rank tests with Holm correction;
* threshold-matched precision (baseline threshold tuned on validation to
  Conf-GCN's recall);
* ablations: uniform, shuffled, inverted confidence; α ∈ {0.5, 2, 4};
  self-loop; k ∈ {5, 10, 20}; description vs feature graph.

## Краткое описание (RU)

Репозиторий магистерской работы: реализация Confidence-Aware GCN, полный
экспериментальный протокол второй статьи и сама статья в LaTeX
(`paper/main.tex`). Все числа в статье генерируются скриптом
`scripts/analyze.py` из файлов `results/runs_*.csv`. Исходный прототип диплома
сохранён в папке `legacy/`.

## License

Code: MIT. Data: CC BY-SA 3.0 (see `data/README.md`).
