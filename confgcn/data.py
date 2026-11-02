"""Data loading, feature construction and graph construction."""
from __future__ import annotations

import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

NUMERIC = ["statuses_count", "followers_count", "friends_count", "favourites_count",
           "account_age_days", "average_tweets_per_day"]
BINARY = ["default_profile", "default_profile_image", "verified", "geo_enabled"]

DEFAULT_SBERT = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------------------------------- #
# Real Twitter data
# --------------------------------------------------------------------------- #
def load_twitter(path: str) -> pd.DataFrame:
    """Load the Twitter human/bot dataset.

    Accepts the original Kaggle/HF CSV (column ``account_type`` in {human, bot})
    or the cleaned parquet redistribution (column ``label`` in {0, 1}).
    """
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "label" not in df.columns:
        df["label"] = (df["account_type"].str.lower() == "bot").astype(int)
    df["description"] = df["description"].fillna("").astype(str)
    for c in BINARY:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1"]).astype(np.float32)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    return df.reset_index(drop=True)


def stratified_subsample(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    parts = [g.sample(n=n_per_class, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def embed_texts(texts: list[str], model_name: str = DEFAULT_SBERT, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def knn_graph(emb: np.ndarray, k: int) -> torch.Tensor:
    """Symmetrised cosine k-NN graph (undirected; both directions stored)."""
    nn_ = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, idx = nn_.kneighbors(emb)
    rows, cols = [], []
    for i, nbrs in enumerate(idx):
        nbrs = [j for j in nbrs if j != i][:k]
        rows += [i] * len(nbrs)
        cols += nbrs
    e = np.array([rows + cols, cols + rows])
    e = np.unique(e, axis=1)
    return torch.as_tensor(e, dtype=torch.long)


@dataclass
class TabularGraph:
    emb: np.ndarray          # (n, 384) description embeddings
    numeric: np.ndarray      # (n, 6) log1p-transformed numeric fields (unscaled)
    binary: np.ndarray       # (n, 4)
    y: np.ndarray            # (n,)
    edge_index: torch.Tensor

    def features(self, train_idx: np.ndarray) -> torch.Tensor:
        """394-d features; numeric fields standardised with training-node statistics only."""
        mu = self.numeric[train_idx].mean(0)
        sd = self.numeric[train_idx].std(0) + 1e-8
        x = np.concatenate([self.emb, (self.numeric - mu) / sd, self.binary], axis=1)
        return torch.as_tensor(x, dtype=torch.float32)


def build_twitter_graph(df: pd.DataFrame, emb: np.ndarray, k: int = 10) -> TabularGraph:
    numeric = np.log1p(df[NUMERIC].to_numpy(dtype=np.float64)).astype(np.float32)
    binary = df[BINARY].to_numpy(dtype=np.float32)
    return TabularGraph(emb=emb, numeric=numeric, binary=binary,
                        y=df["label"].to_numpy(dtype=np.int64), edge_index=knn_graph(emb, k))


def graph_space(emb: np.ndarray, numeric: np.ndarray, binary: np.ndarray, kind: str) -> np.ndarray:
    """Vectors on which the k-NN graph is built.

    ``desc``: description embeddings only (paper default).
    ``meta``: z-scored metadata only.
    ``feat``: description embedding concatenated with z-scored metadata, the
    10-d metadata block scaled by 1/sqrt(10) so that both blocks have
    comparable norm. Standardisation is label-free (unsupervised graph
    construction in the transductive setting).
    """
    if kind == "desc":
        return emb
    meta = np.concatenate([numeric, binary], axis=1).astype(np.float64)
    meta = (meta - meta.mean(0)) / (meta.std(0) + 1e-8)
    if kind == "feat":
        return np.concatenate([emb, meta / np.sqrt(meta.shape[1])], axis=1).astype(np.float32)
    if kind == "meta":
        return meta.astype(np.float32)
    raise ValueError(kind)


def load_processed(path: str, k: int = 10, graph: str = "desc") -> TabularGraph:
    """Load the cached subsample produced by scripts/prepare_data.py."""
    z = np.load(path, allow_pickle=True)
    numeric = np.log1p(z["numeric_raw"]).astype(np.float32)
    space = graph_space(z["emb"], numeric, z["binary"], graph)
    return TabularGraph(emb=z["emb"], numeric=numeric, binary=z["binary"], y=z["y"],
                        edge_index=knn_graph(space, k))


# --------------------------------------------------------------------------- #
# Synthetic social graph (correctness check, RQ1)
# --------------------------------------------------------------------------- #
HUMAN_TOPICS = [
    "Сегодня отличная погода для прогулки в парке",
    "Только что посмотрел новый фильм, очень понравилось",
    "Работа над новым проектом идет полным ходом",
    "Встретил старого друга, было здорово поболтать",
    "Готовлю ужин, пробую новый рецепт",
    "Читаю интересную книгу о путешествиях",
    "Планирую отпуск на следующее лето",
    "Занимаюсь спортом, чувствую себя отлично",
    "Слушаю новую музыку, открываю для себя новые жанры",
    "Обсуждаю с коллегами интересные идеи",
    "Гуляю с собакой, она очень радуется",
    "Изучаю новый язык программирования",
    "Смотрю документальный фильм о природе",
    "Готовлюсь к важной презентации",
    "Встречаюсь с семьей на выходных",
]
BOT_TEMPLATES = [
    "Получите бесплатные биткоины прямо сейчас! Переходите по ссылке",
    "Уникальная возможность заработать на криптовалюте без вложений",
    "Быстрый заработок в интернете через криптовалюту гарантирован",
    "Получите двойной бонус при регистрации на криптобирже",
    "Секретный способ заработка на биткоинах раскрыт",
    "Заработайте тысячи долларов за один день на криптовалюте",
    "Бесплатные криптомонеты ждут вас, регистрируйтесь сейчас",
    "Проверенный метод заработка на криптовалюте без рисков",
    "Получите доступ к эксклюзивной крипто-стратегии",
    "Заработайте миллион на биткоинах за неделю",
]
BOT_VARIATIONS = ["", " Не упустите шанс!", " Ограниченное предложение!", " Только сегодня!",
                  " Проверено тысячами!"]


def synthetic_graph(n_humans=400, n_bots=100, n_clusters=2, n_disguise=20, sim_threshold=0.85,
                    seed=42, model_name: str = DEFAULT_SBERT):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n_humans, 3, seed=seed)
    G.add_nodes_from(range(n_humans, n_humans + n_bots))
    per = n_bots // n_clusters
    for c in range(n_clusters):
        members = range(n_humans + c * per, n_humans + (c + 1) * per)
        G.add_edges_from((a, b) for a in members for b in members if a < b)
    for _ in range(n_disguise):
        G.add_edge(int(rng.integers(n_humans, n_humans + n_bots)), int(rng.integers(0, n_humans)))
    n_struct = G.number_of_edges()

    texts = []
    for _ in range(n_humans):
        a, b = rng.choice(HUMAN_TOPICS, size=2, replace=False)
        texts.append(f"{a}. {b}.")
    for _ in range(n_bots):
        texts.append(str(rng.choice(BOT_TEMPLATES)) + str(rng.choice(BOT_VARIATIONS)))
    emb = embed_texts(texts, model_name)
    sim = emb @ emb.T
    iu = np.triu_indices(len(texts), 1)
    mask = sim[iu] > sim_threshold
    sem = [(int(i), int(j)) for i, j in zip(iu[0][mask], iu[1][mask])]
    before = G.number_of_edges()
    G.add_edges_from(sem)
    n_sem_new = G.number_of_edges() - before

    y = np.array([0] * n_humans + [1] * n_bots, dtype=np.int64)
    e = np.array(G.edges()).T
    edge_index = torch.as_tensor(np.concatenate([e, e[::-1]], axis=1), dtype=torch.long)
    stats = dict(nodes=G.number_of_nodes(), structural_edges=n_struct, semantic_pairs=len(sem),
                 semantic_edges_added=n_sem_new, total_edges=G.number_of_edges())
    return G, emb, y, edge_index, stats
