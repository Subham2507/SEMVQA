import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import hdbscan
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def prefix_of(path):
    fname = os.path.basename(path)
    for p in ("composite_", "ni_alloy_", "ceramics_prod_"):
        if fname.startswith(p):
            return p
    return "img"

def contact_sheet(image_paths, out_path, cols=6, thumb=128):
    rows = (len(image_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, p in enumerate(image_paths):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
        except Exception:
            continue
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="training_data/dinov3_clusters")
    ap.add_argument("--pca-dims", type=int, default=50)
    ap.add_argument("--min-cluster-size", type=int, default=40)
    ap.add_argument("--min-samples", type=int, default=None,
                     help="HDBSCAN density threshold; defaults to min-cluster-size (strict) if unset. "
                          "Lower = looser clusters, less noise.")
    ap.add_argument("--cluster-selection-epsilon", type=float, default=0.0,
                     help="merge clusters closer than this distance; raises this from 0 to pull "
                          "nearby micro-clusters together and reduce noise.")
    ap.add_argument("--samples-per-cluster", type=int, default=18)
    ap.add_argument("--top-outliers", type=int, default=100,
                     help="how many highest-outlier-score images to export as filtering candidates")
    ap.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan",
                     help="hdbscan: density-based, auto-discovers cluster count + flags noise. "
                          "kmeans: forces every point into one of k clusters, no noise concept -- "
                          "better match if the embedding space is a continuum rather than dense blobs.")
    ap.add_argument("--k", type=int, default=None,
                     help="kmeans only: fixed cluster count. If unset, sweeps --k-min..--k-max in "
                          "steps of 5 and picks the k with the best (subsampled) silhouette score.")
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--projection", choices=["umap", "tsne"], default="umap",
                     help="umap scales to tens of thousands of points in ~minutes; sklearn's t-SNE is "
                          "much slower at this size, so tsne mode projects a random subsample instead "
                          "of all points (see --tsne-subsample).")
    ap.add_argument("--tsne-subsample", type=int, default=8000)
    args = ap.parse_args()

    embs = np.load(os.path.join(args.input_dir, "embeddings.npy"))
    with open(os.path.join(args.input_dir, "image_paths.json")) as f:
        paths = json.load(f)
    print(f"{len(paths)} embeddings, dim={embs.shape[1]}")

    embs = normalize(embs)
    reduced = PCA(n_components=args.pca_dims, random_state=42).fit_transform(embs)

    if args.method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(reduced)

        outlier_scores = clusterer.outlier_scores_
    else:
        if args.k:
            k_values = [args.k]
        else:
            k_values = list(range(args.k_min, args.k_max + 1, 5))
        best_k, best_score, best_labels, best_km = None, -1, None, None
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            trial_labels = km.fit_predict(reduced)

            score = silhouette_score(reduced, trial_labels, sample_size=5000, random_state=42)
            print(f"k={k}: silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score, best_labels, best_km = k, score, trial_labels, km
        print(f"Best k={best_k} (silhouette={best_score:.4f})")
        labels = best_labels

        outlier_scores = np.linalg.norm(reduced - best_km.cluster_centers_[labels], axis=1)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters} clusters found, {n_noise} noise/outlier points ({n_noise / len(labels):.1%})")

    top_idx = np.argsort(outlier_scores)[::-1][:args.top_outliers]
    outlier_candidates = [
        {"image": paths[i], "outlier_score": float(outlier_scores[i]), "cluster": int(labels[i])}
        for i in top_idx
    ]
    with open(os.path.join(args.input_dir, "outlier_candidates.json"), "w") as f:
        json.dump(outlier_candidates, f, indent=2)
    contact_sheet([c["image"] for c in outlier_candidates[:60]],
                   os.path.join(args.input_dir, "outlier_top_samples.png"))
    print(f"Wrote top {len(outlier_candidates)} outlier candidates -> outlier_candidates.json, "
          f"outlier_top_samples.png")

    os.makedirs(args.input_dir, exist_ok=True)

    if args.projection == "umap":
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(reduced)
        plot_reduced, plot_labels = reduced, labels
    else:
        from sklearn.manifold import TSNE
        n = min(args.tsne_subsample, len(reduced))
        sub_idx = np.random.RandomState(42).choice(len(reduced), n, replace=False)
        plot_reduced, plot_labels = reduced[sub_idx], labels[sub_idx]
        proj = TSNE(n_components=2, random_state=42, init="pca").fit_transform(plot_reduced)
        print(f"t-SNE projected a random {n}-point subsample (full set would be too slow)")

    plt.figure(figsize=(12, 10))
    noise_mask = plot_labels == -1
    plt.scatter(proj[noise_mask, 0], proj[noise_mask, 1], s=3, c="lightgray", label="noise")
    plt.scatter(proj[~noise_mask, 0], proj[~noise_mask, 1], s=3, c=plot_labels[~noise_mask], cmap="tab20")
    plt.title(f"DINOv3 embeddings, {args.projection.upper()} projection "
              f"({n_clusters} {args.method} clusters, {n_noise} noise pts)")
    plt.savefig(os.path.join(args.input_dir, f"{args.projection}_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()

    cluster_summary = {}
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        cluster_paths = [paths[i] for i in idx]
        prefixes = {}
        for p in cluster_paths:
            pre = prefix_of(p)
            prefixes[pre] = prefixes.get(pre, 0) + 1
        cluster_summary[str(label)] = {
            "size": len(idx),
            "pct_of_dataset": round(len(idx) / len(labels) * 100, 2),
            "by_prefix": prefixes,
            "sample_images": [os.path.basename(p) for p in cluster_paths[:10]],
        }

        if label != -1:
            n = min(args.samples_per_cluster, len(cluster_paths))
            sample = list(np.random.RandomState(42).choice(cluster_paths, n, replace=False))
            contact_sheet(sample, os.path.join(args.input_dir, f"cluster_{label}_samples.png"))

    method_params = ({"min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples,
                       "cluster_selection_epsilon": args.cluster_selection_epsilon}
                      if args.method == "hdbscan" else {"k": best_k, "silhouette": round(best_score, 4)})
    with open(os.path.join(args.input_dir, "cluster_summary.json"), "w") as f:
        json.dump({
            "n_images": len(labels),
            "method": args.method,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(n_noise / len(labels) * 100, 2),
            "method_params": method_params,
            "clusters": cluster_summary,
        }, f, indent=2)

    labels_by_path = {paths[i]: int(labels[i]) for i in range(len(paths))}
    with open(os.path.join(args.input_dir, "cluster_assignments.json"), "w") as f:
        json.dump(labels_by_path, f)

    print(f"Wrote cluster_summary.json, cluster_assignments.json, {args.projection}_clusters.png, "
          f"cluster_<id>_samples.png -> {args.input_dir}")

if __name__ == "__main__":
    main()
