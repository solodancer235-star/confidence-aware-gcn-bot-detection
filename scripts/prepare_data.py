"""Build the 5 000-account stratified subsample and cache SBERT embeddings.

Usage:
    python scripts/prepare_data.py --data data/twitter_human_bots_clean.parquet
    python scripts/prepare_data.py --data path/to/twitter_human_bots_dataset.csv   # original Kaggle CSV
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confgcn.data import BINARY, DEFAULT_SBERT, NUMERIC, embed_texts, load_twitter, stratified_subsample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/twitter_human_bots_clean.parquet")
ap.add_argument("--n-per-class", type=int, default=2500)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--model", default=DEFAULT_SBERT)
ap.add_argument("--out", default="data/processed/twitter_5000.npz")
a = ap.parse_args()

df = load_twitter(a.data)
print(f"full dataset: {len(df)} accounts, humans={int((df.label == 0).sum())}, bots={int((df.label == 1).sum())}")
sub = stratified_subsample(df, a.n_per_class, a.seed)
print(f"subsample: {len(sub)} accounts, empty descriptions: {int((sub.description.str.strip() == '').sum())}")
emb = embed_texts(sub["description"].tolist(), a.model)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez_compressed(a.out, emb=emb, numeric_raw=sub[NUMERIC].to_numpy(np.float64),
                    binary=sub[BINARY].to_numpy(np.float32), y=sub["label"].to_numpy(np.int64),
                    description=np.array(sub["description"].tolist(), dtype=object),
                    full_n=len(df), full_bots=int((df.label == 1).sum()))
print("saved", a.out, emb.shape)
