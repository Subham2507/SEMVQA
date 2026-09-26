import argparse
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

DEFAULT_EMB_DIR = "."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--pca-dims", type=int, default=50)
    ap.add_argument("--silhouette-sample-size", type=int, default=5000)
    args = ap.parse_args()

    embs = np.load(f"{args.embeddings_dir}/embeddings.npy")
    print(f"{embs.shape[0]} embeddings, dim={embs.shape[1]}")

    reduced = PCA(n_components=args.pca_dims, random_state=42).fit_transform(normalize(embs))
    print(f"PCA({args.pca_dims}) done")

    results = []
    for k in range(args.k_min, args.k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels, sample_size=args.silhouette_sample_size, random_state=42)
        db = davies_bouldin_score(reduced, labels)
        ch = calinski_harabasz_score(reduced, labels)
        sizes = np.bincount(labels)
        balance = sizes.min() / sizes.max()
        results.append({
            "k": k, "silhouette": float(sil), "davies_bouldin": float(db),
            "calinski_harabasz": float(ch), "cluster_sizes": sizes.tolist(),
            "size_balance": float(balance),
        })
        print(f"k={k:2d}  silhouette={sil:.4f} (higher better)  "
              f"davies_bouldin={db:.4f} (lower better)  "
              f"calinski_harabasz={ch:8.1f} (higher better)  "
              f"size_balance={balance:.2f}  sizes={sizes.tolist()}")

    out_path = f"{args.embeddings_dir}/kmeans_k_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
