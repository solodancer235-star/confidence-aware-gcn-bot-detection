# Confidence-Aware GCN for Social Bot Detection

Code, experiments and paper for **"Confidence-Aware Graph Neural Networks for Social
Bot Detection: Message-Passing Reweighting and Similarity-Graph Construction"**
(K. Kassymkhan, Astana IT University; L. L. La, L.N. Gumilyov Eurasian National University).

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

**Best detector: GraphSAGE on a metadata k-NN graph — F1 0.840, AUROC 0.921**
(+3.2 F1 over a feature-only MLP, +9.9 F1 over the original GCN-on-descriptions
pipeline, p < 1e-6).

| Model | Description graph F1 | Metadata graph F1 | Feature graph F1 |
|---|---|---|---|
| GCN | 0.740 | 0.826 | 0.820 |
| GraphSAGE | 0.803 | **0.840** | 0.825 |
| GAT | 0.734 | 0.823 | 0.814 |
| Conf-GCN | 0.731 | 0.824 | 0.818 |
| MLP (no graph) | 0.807 | 0.807 | 0.807 |

* **Graph construction is decisive:** description graph (edge homophily 0.62)
  → every GCN-type model is worse than the MLP; metadata graph (0.78) → every
  GNN beats it.
* **Few labels:** with 100 / 250 / 500 labelled accounts GraphSAGE +
  confidence-based self-training is the best model (F1 0.764 / 0.781 / 0.795
  vs MLP 0.744 / 0.761 / 0.773).
* **Confidence reweighting:** the adaptive variants (AC-GCN / AC-SAGE, learnable
  exponent on top of the baseline normalisation) fix the AUROC loss of the
  original Conf-GCN and stay within 0.5 F1 of their baselines, but do not beat
  them; learned exponents shrink from 1 to ≈0.6–0.8. The earlier single-split
  precision gain of Conf-GCN does not replicate.

Full tables: [`results/summary.md`](results/summary.md), [`results/summary.json`](results/summary.json);
paper: [`paper/main.pdf`](paper/main.pdf) (source `paper/main.tex`).

## Repository layout

```
confgcn/            library
  layers.py         ConfidenceWeightedConv, ConfidenceGCNConv, ConfidenceSAGEConv
  models.py         MLP, GCN, GraphSAGE, GAT, ConfidenceAwareGCN, AdaptiveConfidenceGNN (AC-GCN/AC-SAGE/agreement)
  data.py           dataset loading, SBERT features, k-NN graphs, synthetic graph
  train.py          training loop (+ pseudo-label self-training), CV splits, metrics, Wilcoxon + Holm
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
./run_all.sh            # ~5 h on a 2-core CPU
```

or step by step:

```bash
python scripts/prepare_data.py                     # builds data/processed/twitter_5000.npz
python -m pytest -q tests
python scripts/run_synthetic.py
python scripts/run_experiments.py --group select   # hyper-parameter selection (val only)
python scripts/run_experiments.py --group main ablation density graph v2 meta meta2 labels
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
  self-loop; k ∈ {5, 10, 20}; description / metadata / feature graph;
* extensions: learnable α (AC-GCN, AC-SAGE), agreement weights, self-training;
* label-scarce regime: 100 / 250 / 500 labels, 10 splits.

## Краткое описание (RU)

Репозиторий магистерской работы: реализация Confidence-Aware GCN, полный
экспериментальный протокол второй статьи и сама статья в LaTeX
(`paper/main.tex`). Все числа в статье генерируются скриптом
`scripts/analyze.py` из файлов `results/runs_*.csv`. Исходный прототип диплома
сохранён в папке `legacy/`.

## License

Code: MIT. Data: CC BY-SA 3.0 (see `data/README.md`).
