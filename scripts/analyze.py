"""Aggregate runs into the paper's tables and figures (no training here).

Outputs
  results/summary.json, results/summary.md
  paper/tables/*.tex
  paper/figures/fig_pr_curves.pdf, fig_confidence.pdf, fig_threshold.pdf
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confgcn.train import holm, metrics, paired_wilcoxon, recall_matched_threshold  # noqa: E402

R, T, F = "results", "paper/tables", "paper/figures"
os.makedirs(T, exist_ok=True)
os.makedirs(F, exist_ok=True)
runs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/runs_*.csv"))], ignore_index=True)
runs = runs.drop_duplicates(["variant", "seed", "fold"], keep="last").sort_values(["variant", "seed", "fold"])
METRICS = ["acc", "prec", "rec", "f1", "auroc", "aucpr"]
NAMES = {"mlp": "MLP (no graph)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "confgcn": "Conf-GCN (ours)"}
REF = "confgcn"
summary = {}


def get(variant):
    d = runs[runs.variant == variant].sort_values(["seed", "fold"])
    assert len(d) == 25, (variant, len(d))
    return d


def ms(x, digits=3):
    return f"{np.mean(x):.{digits}f}$\\pm${np.std(x, ddof=1):.{digits}f}"


def fmt_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


ref = get(REF)

# ------------------------------------------------------------------ Table V
cmp = ["mlp", "gcn", "sage", "gat"]
pv = {m: {} for m in METRICS}
for m in METRICS:
    raw = {b: paired_wilcoxon(get(b)[f"test_{m}"].values, ref[f"test_{m}"].values) for b in cmp}
    pv[m] = holm(raw)
summary["main"] = {}
lines = []
for v in cmp + [REF]:
    d = get(v)
    summary["main"][v] = {m: dict(mean=float(d[f"test_{m}"].mean()), std=float(d[f"test_{m}"].std(ddof=1)))
                          for m in METRICS}
    summary["main"][v]["p_holm"] = {m: pv[m].get(v) for m in METRICS} if v != REF else None
    cells = [ms(d[f"test_{m}"]) for m in METRICS]
    p1 = fmt_p(pv["f1"][v]) if v != REF else "---"
    p2 = fmt_p(pv["prec"][v]) if v != REF else "---"
    lines.append(f"{NAMES[v]} & " + " & ".join(cells) + f" & {p1} & {p2} \\\\")
with open(f"{T}/table_main.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ Table VI (ablations)
ABL = [("gcn", "GCN (reference)"), ("conf_uniform", "Uniform confidence ($c\\equiv1$)"),
       ("conf_shuffle", "Shuffled confidence"), ("conf_inverted", "Inverted weighting"),
       ("conf_alpha0.5", "$\\alpha=0.5$"), ("conf_alpha2", "$\\alpha=2$"), ("conf_alpha4", "$\\alpha=4$"),
       ("conf_uniform_selfloop", "Uniform conf.\\ + self-loop"), ("conf_selfloop", "Conf-GCN + self-loop"),
       ("confgcn", "Conf-GCN (full, $\\alpha=1$)")]
abl_cmp = [v for v, _ in ABL if v != REF]
pa = holm({v: paired_wilcoxon(get(v).test_f1.values, ref.test_f1.values) for v in abl_cmp})
pa_auc = holm({v: paired_wilcoxon(get(v).test_auroc.values, ref.test_auroc.values) for v in abl_cmp})
summary["ablation"] = {}
lines = []
for v, name in ABL:
    d = get(v)
    summary["ablation"][v] = {m: dict(mean=float(d[f"test_{m}"].mean()), std=float(d[f"test_{m}"].std(ddof=1)))
                              for m in METRICS}
    summary["ablation"][v]["p_f1_vs_full"] = pa.get(v)
    summary["ablation"][v]["p_auroc_vs_full"] = pa_auc.get(v)
    cells = [ms(d[f"test_{m}"]) for m in ["prec", "rec", "f1", "auroc"]]
    lines.append(f"{name} & " + " & ".join(cells) + f" & {fmt_p(pa[v]) if v in pa else '---'} \\\\")
with open(f"{T}/table_ablation.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ density
summary["density"] = {}
lines = []
for model in ["gcn", "confgcn"]:
    row = [NAMES[model]]
    for k in (5, 10, 20):
        v = model if k == 10 else f"{model}_k{k}"
        d = get(v)
        summary["density"][f"{model}_k{k}"] = {m: float(d[f"test_{m}"].mean()) for m in METRICS}
        row += [ms(d.test_f1), ms(d.test_auroc)]
    lines.append(" & ".join(row) + " \\\\")
for k in (5, 10, 20):
    a_ = get("gcn" if k == 10 else f"gcn_k{k}")
    b_ = get("confgcn" if k == 10 else f"confgcn_k{k}")
    summary["density"][f"p_f1_k{k}"] = paired_wilcoxon(a_.test_f1.values, b_.test_f1.values)
with open(f"{T}/table_density.tex", "w") as f:
    f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ graph construction
summary["graph"] = {}
lines = []
for model in ["gcn", "sage", "gat", "confgcn"]:
    a_, b_ = get(model), get(f"{model}_featgraph")
    summary["graph"][model] = dict(
        desc={m: float(a_[f"test_{m}"].mean()) for m in METRICS},
        feat={m: float(b_[f"test_{m}"].mean()) for m in METRICS},
        p_f1=paired_wilcoxon(a_.test_f1.values, b_.test_f1.values),
        p_f1_vs_mlp_feat=paired_wilcoxon(b_.test_f1.values, get("mlp").test_f1.values))
    lines.append(f"{NAMES[model]} & {ms(a_.test_f1)} & {ms(a_.test_auroc)} & {ms(b_.test_f1)} & {ms(b_.test_auroc)} \\\\")
mlp_ = get("mlp")
lines.append(f"\\midrule\n{NAMES['mlp']} & \\multicolumn{{4}}{{c}}{{F1 {ms(mlp_.test_f1)}, AUROC {ms(mlp_.test_auroc)}}} \\\\")
fc, fg = get("confgcn_featgraph"), get("gcn_featgraph")
summary["graph"]["feat_conf_vs_gcn"] = {m: paired_wilcoxon(fc[f"test_{m}"].values, fg[f"test_{m}"].values)
                                        for m in METRICS}
with open(f"{T}/table_graph.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ threshold-matched analysis
def probs(variant):
    out = []
    for _, r in get(variant).iterrows():
        out.append(np.load(f"{R}/probs/{variant}/{int(r.seed)}_{int(r.fold)}.npz"))
    return out


P = {v: probs(v) for v in cmp + [REF]}
tm = {}
for b in cmp:
    prec_b, rec_b, prec_c, rec_c = [], [], [], []
    for zc, zb in zip(P[REF], P[b]):
        target = metrics(zc["y_val"], zc["p_val"])["rec"]
        thr = recall_matched_threshold(zb["y_val"], zb["p_val"], target)
        mb = metrics(zb["y_test"], zb["p_test"], thr)
        mc = metrics(zc["y_test"], zc["p_test"])
        prec_b.append(mb["prec"]); rec_b.append(mb["rec"])
        prec_c.append(mc["prec"]); rec_c.append(mc["rec"])
    tm[b] = dict(prec_baseline=np.array(prec_b), rec_baseline=np.array(rec_b),
                 prec_conf=np.array(prec_c), rec_conf=np.array(rec_c))
ptm = holm({b: paired_wilcoxon(tm[b]["prec_baseline"], tm[b]["prec_conf"]) for b in cmp})
summary["threshold_matched"] = {}
lines = []
for b in cmp:
    t = tm[b]
    summary["threshold_matched"][b] = dict(
        prec_baseline=float(t["prec_baseline"].mean()), rec_baseline=float(t["rec_baseline"].mean()),
        prec_conf=float(t["prec_conf"].mean()), rec_conf=float(t["rec_conf"].mean()),
        delta_prec=float((t["prec_conf"] - t["prec_baseline"]).mean()), p_holm=ptm[b])
    lines.append(f"{NAMES[b]} & {ms(t['rec_baseline'])} & {ms(t['prec_baseline'])} & {ms(t['prec_conf'])} & "
                 f"{(t['prec_conf'] - t['prec_baseline']).mean():+.3f} & {fmt_p(ptm[b])} \\\\")
with open(f"{T}/table_threshold.tex", "w") as f:
    f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ calibration (ECE)
def ece(y, p, bins=10):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


summary["ece"] = {v: float(np.mean([ece(z["y_test"], z["p_test"]) for z in P[v]])) for v in P}

# ------------------------------------------------------------------ PR curves
grid = np.linspace(0, 1, 201)
fig, ax = plt.subplots(figsize=(3.5, 2.8))
styles = {"mlp": ":", "gcn": "--", "sage": "-.", "gat": (0, (3, 1, 1, 1)), "confgcn": "-"}
for v in cmp + [REF]:
    curves = []
    for z in P[v]:
        pr, rc, _ = precision_recall_curve(z["y_test"], z["p_test"])
        order = np.argsort(rc)
        rc_s, pr_s = rc[order], np.maximum.accumulate(pr[order][::-1])[::-1]
        curves.append(np.interp(grid, rc_s, pr_s))
    ax.plot(grid, np.mean(curves, 0), linestyle=styles[v], lw=1.4 if v == REF else 1.0, label=NAMES[v])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.45, 1.01); ax.set_xlim(0, 1)
ax.grid(alpha=0.3); ax.legend(fontsize=6.5, frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig(f"{F}/fig_pr_curves.pdf"); plt.close()

# ------------------------------------------------------------------ confidence and loss curves
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
cc = np.array([json.loads(s) for s in ref.conf_curve])
axs[0].plot(np.arange(1, cc.shape[1] + 1), cc.mean(0), label="Twitter (Conf-GCN)")
axs[0].fill_between(np.arange(1, cc.shape[1] + 1), cc.min(0), cc.max(0), alpha=0.2)
syn_path = f"{R}/synthetic/synthetic_results.json"
if os.path.exists(syn_path):
    syn = json.load(open(syn_path))
    sc = syn["single"]["confgcn"]["conf"]
    axs[0].plot(np.arange(1, len(sc) + 1), sc, "--", label="Synthetic (Conf-GCN)")
    summary["synthetic"] = dict(stats=syn["stats"], gcn=syn["single"]["gcn"]["test"],
                                confgcn=syn["single"]["confgcn"]["test"],
                                conf_epoch20=sc[19], conf_epoch100=sc[-1],
                                loss_final_gcn=syn["single"]["gcn"]["loss"][-1],
                                loss_final_conf=syn["single"]["confgcn"]["loss"][-1])
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Mean confidence"); axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=7, frameon=False)
for v in ["gcn", REF]:
    lc = np.array([json.loads(s) for s in get(v).loss_curve])
    axs[1].plot(np.arange(1, lc.shape[1] + 1), lc.mean(0), styles[v], label=NAMES[v])
axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Training loss"); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f"{F}/fig_confidence.pdf"); plt.close()
summary["conf_curve_real"] = dict(epoch1=float(cc.mean(0)[0]), epoch20=float(cc.mean(0)[19]),
                                  epoch100=float(cc.mean(0)[-1]))

# ------------------------------------------------------------------ hyper-parameters, cost
summary["hparams"] = json.load(open(f"{R}/selected_hparams.json"))
summary["secs_per_run"] = {v: float(get(v).secs.mean()) for v in cmp + [REF]}
syn_cv = f"{R}/synthetic/cv_runs.csv"
if os.path.exists(syn_cv):
    d = pd.read_csv(syn_cv)
    summary["synthetic_cv"] = d.groupby("model")[["acc", "prec", "rec", "f1"]].mean().round(4).to_dict("index")


def conv(o):
    if isinstance(o, dict):
        return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(conv(summary), open(f"{R}/summary.json", "w"), indent=1)

with open(f"{R}/summary.md", "w") as f:
    f.write("# Results summary (5-fold CV x 5 seeds, mean +- std)\n\n## Main comparison\n\n")
    f.write("| Model | " + " | ".join(METRICS) + " | p(F1) | p(Prec) |\n|" + "---|" * (len(METRICS) + 3) + "\n")
    for v in cmp + [REF]:
        s = summary["main"][v]
        p = summary["main"][v]["p_holm"]
        f.write(f"| {NAMES[v]} | " + " | ".join(f"{s[m]['mean']:.3f}+-{s[m]['std']:.3f}" for m in METRICS)
                + (f" | {p['f1']:.4f} | {p['prec']:.4f} |\n" if p else " | - | - |\n"))
    f.write("\n## Ablations\n\n| Variant | prec | rec | f1 | auroc | p(F1 vs full) |\n|---|---|---|---|---|---|\n")
    for v, name in ABL:
        s = summary["ablation"][v]
        p = s["p_f1_vs_full"]
        f.write(f"| {v} | " + " | ".join(f"{s[m]['mean']:.3f}" for m in ["prec", "rec", "f1", "auroc"])
                + f" | {'-' if p is None else f'{p:.4f}'} |\n")
    f.write("\n## Threshold-matched precision (baseline threshold tuned on val to Conf-GCN's val recall)\n\n")
    f.write("| Baseline | rec_baseline | prec_baseline | prec_conf | delta | p |\n|---|---|---|---|---|---|\n")
    for b in cmp:
        s = summary["threshold_matched"][b]
        f.write(f"| {b} | {s['rec_baseline']:.3f} | {s['prec_baseline']:.3f} | {s['prec_conf']:.3f} | "
                f"{s['delta_prec']:+.3f} | {s['p_holm']:.4f} |\n")
    f.write(f"\n## ECE (test, 10 bins)\n\n{json.dumps(summary['ece'], indent=1)}\n")
    f.write(f"\n## Selected hyper-parameters\n\n{json.dumps(summary['hparams'], indent=1)}\n")
print(open(f"{R}/summary.md").read())
