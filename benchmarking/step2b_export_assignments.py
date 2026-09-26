import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

DEFAULT_EMB_DIR = "."
K = 5

def prefix_of(path):
    fname = os.path.basename(path)
    for p in ("composite_", "ni_alloy_", "ceramics_prod_"):
        if fname.startswith(p):
            return p
    return "img"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--pca-dims", type=int, default=50)
    ap.add_argument("--thumb", type=int, default=110)
    args = ap.parse_args()
    D = args.embeddings_dir

    embs = np.load(f"{D}/embeddings.npy")
    paths = json.load(open(f"{D}/image_paths.json"))
    reduced = PCA(n_components=args.pca_dims, random_state=42).fit_transform(normalize(embs))

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(reduced)
    centroids = km.cluster_centers_

    existing_assign = json.load(open(f"{D}/cluster_assignments.json"))
    existing_labels = np.array([existing_assign[p] for p in paths])
    match_frac = float((labels == existing_labels).mean())
    print(f"Reproducibility check: {match_frac:.1%} identical to existing cluster_assignments.json")

    np.save(f"{D}/cluster_centroids.npy", centroids)
    print(f"Saved -> {D}/cluster_centroids.npy  (shape {centroids.shape}, PCA(50) space)")

    all_rows = []
    for c in range(K):
        idx = np.where(labels == c)[0]
        dists = np.linalg.norm(reduced[idx] - centroids[c], axis=1)
        order = np.argsort(dists)
        n = len(order)
        p10, p70 = int(np.ceil(n * 0.10)), int(np.ceil(n * 0.70))
        band_of_rank = {}
        for rank in range(n):
            band_of_rank[rank] = "core" if rank < p10 else ("mid" if rank < p70 else "peripheral")
        for rank, oi in enumerate(order):
            gi = idx[oi]
            all_rows.append({
                "image": paths[gi], "filename": os.path.basename(paths[gi]),
                "prefix": prefix_of(paths[gi]), "cluster_id": int(c), "cluster_size": int(n),
                "distance_to_centroid": float(dists[oi]), "band": band_of_rank[rank],
                "rank_within_cluster": rank + 1, "percentile_within_cluster": round((rank + 1) / n * 100, 4),
            })
    all_rows.sort(key=lambda r: (r["cluster_id"], r["rank_within_cluster"]))
    with open(f"{D}/all_cluster_assignments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved -> {D}/all_cluster_assignments.csv  ({len(all_rows)} rows)")

    rep_path = f"{D}/representative_subset_100_with_captions.json"
    rep = json.load(open(rep_path))
    with open(f"{D}/representative_100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
        w.writeheader()
        w.writerows(rep)
    print(f"Saved -> {D}/representative_100.csv  ({len(rep)} rows)")

    sweep_path = f"{D}/kmeans_k_sweep.json"
    if os.path.exists(sweep_path):
        sweep = json.load(open(sweep_path))
        with open(f"{D}/cluster_metrics.csv", "w", newline="") as f:
            fieldnames = ["k", "silhouette", "davies_bouldin", "calinski_harabasz", "size_balance", "cluster_sizes"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in sweep:
                r = dict(row)
                r["cluster_sizes"] = ";".join(str(x) for x in r["cluster_sizes"])
                w.writerow({k: r[k] for k in fieldnames})
        print(f"Saved -> {D}/cluster_metrics.csv  ({len(sweep)} rows)")
    else:
        print(f"SKIPPED cluster_metrics.csv -- {sweep_path} not found, run sweep_kmeans_k.py first")

    thumb = args.thumb
    pad = 4
    label_h = 18
    cols_per_band = 7
    band_order = ["core", "mid", "peripheral"]
    by_cluster = {c: {b: [] for b in band_order} for c in range(K)}
    for r in rep:
        by_cluster[r["cluster_id"]][r["sampling_category"]].append(r)
    for c in by_cluster:
        for b in by_cluster[c]:
            by_cluster[c][b].sort(key=lambda r: r["rank_within_cluster"])

    band_width = cols_per_band * (thumb + pad)
    sheet_w = 3 * band_width + 2 * 20
    row_h = thumb + pad + label_h
    sheet_h = K * row_h + (K + 1) * 10 + 30

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    y = 10
    draw.text((10, y), "Representative subset (100 images): Core | Mid | Peripheral per cluster", fill="black", font=font)
    y += 22
    for c in range(K):
        x = 0
        for bi, band in enumerate(band_order):
            draw.text((x + 5, y), f"cluster {c} - {band}", fill="black", font=font)
            imgs_y = y + label_h
            for j, rec in enumerate(by_cluster[c][band]):
                try:
                    im = Image.open(rec["image"]).convert("RGB").resize((thumb, thumb))
                except Exception:
                    im = Image.new("RGB", (thumb, thumb), "gray")
                sheet.paste(im, (x + j * (thumb + pad), imgs_y))
            x += band_width + 20
        y += row_h + 10

    out_path = f"{D}/representative_contact_sheet.png"
    sheet.save(out_path)
    print(f"Saved -> {out_path}  ({sheet_w}x{sheet_h})")

if __name__ == "__main__":
    main()
