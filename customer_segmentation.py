"""K-Means diagnostics, final clustering, and cluster business labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = ["Recency", "Frequency", "Monetary"]


def prepare_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = rfm[FEATURES].copy()
    # Frequency and Monetary have long right tails. log1p preserves zeros if present.
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])
    scaled = StandardScaler().fit_transform(features)
    return features, scaled


def cluster_diagnostics(scaled: np.ndarray, candidates=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette_score": silhouette_score(scaled, labels)})
    return pd.DataFrame(rows)


def choose_k(diagnostics: pd.DataFrame) -> int:
    """Objective selection: highest silhouette among the tested candidate values."""
    return int(diagnostics.loc[diagnostics["silhouette_score"].idxmax(), "k"])


def _label_clusters(summary: pd.DataFrame) -> dict[int, str]:
    """Rank clusters by observed recency, frequency, and spend for readable names."""
    work = summary.copy()
    work["value_rank"] = work["AvgMonetary"].rank(method="first", ascending=False)
    work["recency_rank"] = work["AvgRecency"].rank(method="first", ascending=True)
    work["frequency_rank"] = work["AvgFrequency"].rank(method="first", ascending=False)
    labels = {}
    for row in work.itertuples():
        if row.value_rank == 1 and row.recency_rank <= 2:
            name = "High-Value Active"
        elif row.recency_rank == 1 and row.frequency_rank <= 2:
            name = "Frequent Recent Buyers"
        elif row.recency_rank == len(work):
            name = "Dormant Customers"
        elif row.value_rank <= 2:
            name = "High-Spend Watchlist"
        else:
            name = "Occasional Customers"
        labels[int(row.Cluster)] = name
    # make labels unique without disguising behavior
    used = {}
    for cluster, name in labels.items():
        used[name] = used.get(name, 0) + 1
        if used[name] > 1:
            labels[cluster] = f"{name} {used[name]}"
    return labels


def run_clustering(rfm: pd.DataFrame, final_k: int):
    _, scaled = prepare_features(rfm)
    model = KMeans(n_clusters=final_k, random_state=42, n_init=20)
    output = rfm.copy()
    output["Cluster"] = model.fit_predict(scaled)
    summary = output.groupby("Cluster", as_index=False).agg(
        Customers=("CustomerID", "nunique"), AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"), AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    )
    summary["CustomerPct"] = summary["Customers"] / summary["Customers"].sum()
    summary["RevenuePct"] = summary["TotalRevenue"] / summary["TotalRevenue"].sum()
    labels = _label_clusters(summary)
    output["Segment"] = output["Cluster"].map(labels)
    summary["Segment"] = summary["Cluster"].map(labels)
    summary = summary[["Cluster", "Segment", "Customers", "AvgRecency", "AvgFrequency", "AvgMonetary", "TotalRevenue", "CustomerPct", "RevenuePct"]]
    pca = PCA(n_components=2, random_state=42).fit(scaled)
    pca_data = pca.transform(scaled)
    output["PCA1"], output["PCA2"] = pca_data[:, 0], pca_data[:, 1]
    perplexity = min(30, max(5, (len(output) - 1) // 3))
    tsne_data = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(scaled)
    output["TSNE1"], output["TSNE2"] = tsne_data[:, 0], tsne_data[:, 1]
    return output, summary, pca.explained_variance_ratio_, model
