import argparse
import csv
import json
import os
import random
from collections import defaultdict

DEFAULT_EMB_DIR = "."
DEFAULT_EXCLUDE = "representative_100.csv"
DEFAULT_QA_SOURCE = "sem_vqa_corpus.jsonl"
K = 5
BANDS = ["core", "mid", "peripheral"]
PER_CELL = 20
SEED = 42

def load_assignments(emb_dir):
    rows = []
    with open(f"{emb_dir}/all_cluster_assignments.csv") as f:
        for r in csv.DictReader(f):
            r["cluster_id"] = int(r["cluster_id"])
            r["distance_to_centroid"] = float(r["distance_to_centroid"])
            rows.append(r)
    return rows

def load_excluded(exclude_path):
    excluded = set()
    if exclude_path.endswith(".csv"):
        with open(exclude_path) as f:
            for r in csv.DictReader(f):
                excluded.add(r["filename"])
    else:
        for r in json.load(open(exclude_path)):
            excluded.add(r["filename"])
    return excluded

def load_qa_metadata(qa_path):
    by_filename = defaultdict(lambda: {"levels": set(), "features": set(), "n_questions": 0})
    with open(qa_path) as f:
        for line in f:
            r = json.loads(line)
            fn = r["filename"]
            entry = by_filename[fn]
            level = (r.get("level") or "").strip().lower()
            feature = (r.get("feature") or "").strip()
            if level:
                entry["levels"].add(level)
            if feature:
                entry["features"].add(feature)
            entry["n_questions"] += 1
    return by_filename

def farthest_point_gap(distance, selected_distances):
    if not selected_distances:
        return float("inf")
    return min(abs(distance - d) for d in selected_distances)

def select_cell(candidates, per_cell, rng):
    remaining = list(candidates)
    covered_levels, covered_features = set(), set()
    selected, reasons = [], []
    selected_distances = []

    while remaining and len(selected) < per_cell:
        best_novelty = -1
        best_candidates = []
        for c in remaining:
            new_levels = c["levels"] - covered_levels
            new_features = c["features"] - covered_features
            novelty = len(new_levels) + len(new_features)
            if novelty > best_novelty:
                best_novelty = novelty
                best_candidates = [c]
            elif novelty == best_novelty:
                best_candidates.append(c)

        if best_novelty > 0:

            best_gap = -1
            spread_tied = []
            for c in best_candidates:
                gap = farthest_point_gap(c["distance_to_centroid"], selected_distances)
                if gap > best_gap:
                    best_gap = gap
                    spread_tied = [c]
                elif gap == best_gap:
                    spread_tied.append(c)
            pick = rng.choice(spread_tied) if len(spread_tied) > 1 else spread_tied[0]
            new_levels = pick["levels"] - covered_levels
            new_features = pick["features"] - covered_features
            reason = f"new coverage: +{len(new_levels)} level(s) {sorted(new_levels)}, +{len(new_features)} feature(s) {sorted(new_features)[:5]}"
            if len(new_features) > 5:
                reason += f" (+{len(new_features) - 5} more)"
            if len(spread_tied) > 1:
                reason += f"; distance-spread tiebreak among {len(spread_tied)} equally-novel candidates"
            covered_levels |= new_levels
            covered_features |= new_features
        else:

            best_gap = -1
            spread_tied = []
            for c in remaining:
                gap = farthest_point_gap(c["distance_to_centroid"], selected_distances)
                if gap > best_gap:
                    best_gap = gap
                    spread_tied = [c]
                elif gap == best_gap:
                    spread_tied.append(c)
            pick = rng.choice(spread_tied) if len(spread_tied) > 1 else spread_tied[0]
            reason = f"distance-diversity fill (coverage exhausted); gap-to-nearest-selected={best_gap:.3f}"
            if len(spread_tied) > 1:
                reason += f"; random tiebreak among {len(spread_tied)} equally-spread candidates"

        selected.append(pick)
        reasons.append(reason)
        selected_distances.append(pick["distance_to_centroid"])
        remaining.remove(pick)

    return selected, reasons

def quantile_stats(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None, "mean": None}

    def interp(p):
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return v[lo] + (v[hi] - v[lo]) * frac

    return {
        "n": n, "min": v[0], "q1": interp(0.25), "median": interp(0.5),
        "q3": interp(0.75), "max": v[-1], "mean": sum(v) / n,
    }

UNDERREPRESENTED_RATIO = 0.5

def generate_coverage_report(final_records, pool, output_prefix):
    n_selected = len(final_records)
    levels_all = ["observation", "detection", "identification", "interpretation"]

    cluster_cov = []
    for c in range(K):
        n = sum(1 for r in final_records if r["cluster_id"] == c)
        cluster_cov.append({"cluster_id": c, "count": n, "pct_of_300": round(n / n_selected * 100, 2)})

    centrality_cov = []
    for b in BANDS:
        n = sum(1 for r in final_records if r["centrality"] == b)
        centrality_cov.append({"centrality": b, "count": n, "pct_of_300": round(n / n_selected * 100, 2)})

    cxc_grid = []
    for c in range(K):
        for b in BANDS:
            n = sum(1 for r in final_records if r["cluster_id"] == c and r["centrality"] == b)
            cxc_grid.append({"cluster_id": c, "centrality": b, "count": n})

    level_cov = []
    for lv in levels_all:
        n = sum(1 for r in final_records if lv in r["question_levels"])
        level_cov.append({"question_level": lv, "n_images": n, "pct_of_300": round(n / n_selected * 100, 2)})

    feature_image_counts = defaultdict(int)
    for r in final_records:
        for ft in r["features"]:
            feature_image_counts[ft] += 1
    feature_cov = [{"feature": ft, "n_images": n, "pct_of_300": round(n / n_selected * 100, 2)}
                   for ft, n in sorted(feature_image_counts.items(), key=lambda kv: -kv[1])]
    n_unique_features_selected = len(feature_image_counts)

    cxc_level = []
    for c in range(K):
        for b in BANDS:
            cell = [r for r in final_records if r["cluster_id"] == c and r["centrality"] == b]
            for lv in levels_all:
                n = sum(1 for r in cell if lv in r["question_levels"])
                cxc_level.append({"cluster_id": c, "centrality": b, "question_level": lv, "n_images": n})

    cxc_feature = {}
    for c in range(K):
        for b in BANDS:
            cell = [r for r in final_records if r["cluster_id"] == c and r["centrality"] == b]
            cell_features = defaultdict(int)
            for r in cell:
                for ft in r["features"]:
                    cell_features[ft] += 1
            cxc_feature[f"cluster_{c}_{b}"] = {
                "n_unique_features": len(cell_features),
                "features": dict(sorted(cell_features.items(), key=lambda kv: -kv[1])),
            }

    dist_by_band = [{"group": b, **quantile_stats([r["distance_to_centroid"] for r in final_records if r["centrality"] == b])}
                     for b in BANDS]
    dist_by_cluster = [{"group": c, **quantile_stats([r["distance_to_centroid"] for r in final_records if r["cluster_id"] == c])}
                        for c in range(K)]
    dist_by_cxc = [{"cluster_id": c, "centrality": b,
                     **quantile_stats([r["distance_to_centroid"] for r in final_records
                                        if r["cluster_id"] == c and r["centrality"] == b])}
                    for c in range(K) for b in BANDS]

    n_pool = len(pool)
    level_pool_pct = {lv: sum(1 for c in pool if lv in c["levels"]) / n_pool * 100 for lv in levels_all}
    level_selected_pct = {row["question_level"]: row["pct_of_300"] for row in level_cov}

    pool_feature_counts = defaultdict(int)
    for c in pool:
        for ft in c["features"]:
            pool_feature_counts[ft] += 1
    top_features_for_comparison = sorted(pool_feature_counts.items(), key=lambda kv: -kv[1])[:30]

    full_vs_selected = {"note": "Descriptive comparison only -- not a statistical significance test.",
                         "n_full_pool": n_pool, "n_selected": n_selected, "levels": [], "top_features": []}
    flags = []
    for lv in levels_all:
        pool_pct = round(level_pool_pct[lv], 2)
        sel_pct = level_selected_pct.get(lv, 0.0)
        row = {"question_level": lv, "full_pool_pct": pool_pct, "selected_pct": sel_pct}
        full_vs_selected["levels"].append(row)
        if sel_pct == 0.0:
            flags.append(f"ABSENT: question level '{lv}' has 0 images in the selected 300 "
                          f"(full pool: {pool_pct}%)")
        elif sel_pct < pool_pct * UNDERREPRESENTED_RATIO:
            flags.append(f"UNDERREPRESENTED: question level '{lv}' is {sel_pct}% of selected "
                          f"vs {pool_pct}% of full pool (< {UNDERREPRESENTED_RATIO:.0%} of pool rate)")

    for ft, pool_n in top_features_for_comparison:
        pool_pct = round(pool_n / n_pool * 100, 3)
        sel_n = feature_image_counts.get(ft, 0)
        sel_pct = round(sel_n / n_selected * 100, 3)
        full_vs_selected["top_features"].append({"feature": ft, "full_pool_pct": pool_pct, "selected_pct": sel_pct})
        if sel_n == 0:
            flags.append(f"ABSENT: feature '{ft}' (top-30 in full pool at {pool_pct}%) has 0 images "
                          f"in the selected 300")
        elif sel_pct < pool_pct * UNDERREPRESENTED_RATIO:
            flags.append(f"UNDERREPRESENTED: feature '{ft}' is {sel_pct}% of selected vs "
                          f"{pool_pct}% of full pool (< {UNDERREPRESENTED_RATIO:.0%} of pool rate)")

    def write_csv(path, rows, fieldnames):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved -> {path}")

    write_csv(f"{output_prefix}_coverage_cluster.csv", cluster_cov, ["cluster_id", "count", "pct_of_300"])
    write_csv(f"{output_prefix}_coverage_centrality.csv", centrality_cov, ["centrality", "count", "pct_of_300"])
    write_csv(f"{output_prefix}_coverage_cluster_x_centrality.csv", cxc_grid, ["cluster_id", "centrality", "count"])
    write_csv(f"{output_prefix}_coverage_question_level.csv", level_cov, ["question_level", "n_images", "pct_of_300"])
    write_csv(f"{output_prefix}_coverage_feature.csv", feature_cov, ["feature", "n_images", "pct_of_300"])
    write_csv(f"{output_prefix}_coverage_cluster_x_centrality_x_level.csv", cxc_level,
              ["cluster_id", "centrality", "question_level", "n_images"])
    write_csv(f"{output_prefix}_coverage_distance_by_centrality.csv", dist_by_band,
              ["group", "n", "min", "q1", "median", "q3", "max", "mean"])
    write_csv(f"{output_prefix}_coverage_distance_by_cluster.csv", dist_by_cluster,
              ["group", "n", "min", "q1", "median", "q3", "max", "mean"])
    write_csv(f"{output_prefix}_coverage_distance_by_cluster_x_centrality.csv", dist_by_cxc,
              ["cluster_id", "centrality", "n", "min", "q1", "median", "q3", "max", "mean"])

    with open(f"{output_prefix}_coverage_cluster_x_centrality_x_feature.json", "w") as f:
        json.dump(cxc_feature, f, indent=2)
    print(f"Saved -> {output_prefix}_coverage_cluster_x_centrality_x_feature.json")

    with open(f"{output_prefix}_coverage_full_vs_selected.json", "w") as f:
        json.dump(full_vs_selected, f, indent=2)
    print(f"Saved -> {output_prefix}_coverage_full_vs_selected.json")

    with open(f"{output_prefix}_coverage_flags.json", "w") as f:
        json.dump({"underrepresented_ratio_threshold": UNDERREPRESENTED_RATIO, "flags": flags}, f, indent=2)
    print(f"Saved -> {output_prefix}_coverage_flags.json  ({len(flags)} flag(s))")

    lines = []
    lines.append("# rep300 subset -- coverage report\n")
    lines.append(f"Selected: {n_selected} images. Full eligible candidate pool: {n_pool} images "
                  f"(65,900 total minus the 100 excluded prior-subset images).\n")

    lines.append("## Cluster coverage\n")
    lines.append("| Cluster | Count | % of 300 |\n|---|---|---|")
    for row in cluster_cov:
        lines.append(f"| C{row['cluster_id']} | {row['count']} | {row['pct_of_300']}% |")

    lines.append("\n## Centrality coverage\n")
    lines.append("| Centrality | Count | % of 300 |\n|---|---|---|")
    for row in centrality_cov:
        lines.append(f"| {row['centrality']} | {row['count']} | {row['pct_of_300']}% |")

    lines.append("\n## Cluster x Centrality (should all be 20)\n")
    header = "| Cluster | " + " | ".join(BANDS) + " |"
    lines.append(header)
    lines.append("|---" * (len(BANDS) + 1) + "|")
    for c in range(K):
        row_counts = [str(next(r["count"] for r in cxc_grid if r["cluster_id"] == c and r["centrality"] == b))
                      for b in BANDS]
        lines.append(f"| C{c} | " + " | ".join(row_counts) + " |")

    lines.append("\n## Question-level coverage (multi-label -- percentages do not sum to 100%)\n")
    lines.append("| Level | Images | % of 300 |\n|---|---|---|")
    for row in level_cov:
        lines.append(f"| {row['question_level']} | {row['n_images']} | {row['pct_of_300']}% |")

    lines.append(f"\n## Feature coverage\n\n{n_unique_features_selected} unique features covered "
                  f"across the 300 (out of {len(pool_feature_counts)} in the full eligible pool). "
                  f"Full per-feature table in `{os.path.basename(output_prefix)}_coverage_feature.csv`. "
                  f"Top 15 shown here:\n")
    lines.append("| Feature | Images | % of 300 |\n|---|---|---|")
    for row in feature_cov[:15]:
        lines.append(f"| {row['feature']} | {row['n_images']} | {row['pct_of_300']}% |")

    lines.append("\n## Centroid-distance stats by centrality\n")
    lines.append("| Group | n | min | Q1 | median | Q3 | max | mean |\n|---|---|---|---|---|---|---|---|")
    for row in dist_by_band:
        lines.append(f"| {row['group']} | {row['n']} | {row['min']:.3f} | {row['q1']:.3f} | "
                      f"{row['median']:.3f} | {row['q3']:.3f} | {row['max']:.3f} | {row['mean']:.3f} |")

    lines.append("\n## Centroid-distance stats by cluster\n")
    lines.append("| Cluster | n | min | Q1 | median | Q3 | max | mean |\n|---|---|---|---|---|---|---|---|")
    for row in dist_by_cluster:
        lines.append(f"| C{row['group']} | {row['n']} | {row['min']:.3f} | {row['q1']:.3f} | "
                      f"{row['median']:.3f} | {row['q3']:.3f} | {row['max']:.3f} | {row['mean']:.3f} |")

    lines.append("\n## Full candidate pool vs. selected 300 (descriptive comparison only -- not a statistical test)\n")
    lines.append("### Question levels\n")
    lines.append("| Level | Full pool % | Selected % |\n|---|---|---|")
    for row in full_vs_selected["levels"]:
        lines.append(f"| {row['question_level']} | {row['full_pool_pct']}% | {row['selected_pct']}% |")
    lines.append("\n### Top-30 features in the full pool, compared to selected\n")
    lines.append("| Feature | Full pool % | Selected % |\n|---|---|---|")
    for row in full_vs_selected["top_features"]:
        lines.append(f"| {row['feature']} | {row['full_pool_pct']}% | {row['selected_pct']}% |")

    lines.append(f"\n## Flags -- absent or highly underrepresented categories "
                 f"(selected rate < {UNDERREPRESENTED_RATIO:.0%} of full-pool rate)\n")
    if flags:
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("None -- no category fell below the underrepresentation threshold.")

    summary_path = f"{output_prefix}_coverage_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved -> {summary_path}")

    print(f"\n=== Coverage flags ({len(flags)}) ===")
    for flag in flags:
        print(f"  {flag}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default=DEFAULT_EMB_DIR)
    ap.add_argument("--exclude-file", default=DEFAULT_EXCLUDE)
    ap.add_argument("--qa-source", default=DEFAULT_QA_SOURCE)
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--output-prefix", default=f"{DEFAULT_EMB_DIR}/representative_subset_300")
    args = ap.parse_args()

    assignments = load_assignments(args.embeddings_dir)
    excluded = load_excluded(args.exclude_file)
    qa_meta = load_qa_metadata(args.qa_source)
    print(f"{len(assignments)} total images, {len(excluded)} excluded (prior 100-subset), "
          f"QA metadata for {len(qa_meta)} filenames")

    no_qa = 0
    pool = []
    for r in assignments:
        if r["filename"] in excluded:
            continue
        meta = qa_meta.get(r["filename"])
        if meta is None:
            no_qa += 1
            continue
        pool.append({
            **r,
            "levels": meta["levels"],
            "features": meta["features"],
            "n_questions": meta["n_questions"],
        })
    print(f"{len(pool)} eligible candidates ({no_qa} skipped -- no QA metadata found)")

    rng = random.Random(SEED)
    final_records = []
    shortfalls = []
    for cluster_id in range(K):
        for band in BANDS:
            cell_candidates = [c for c in pool if c["cluster_id"] == cluster_id and c["band"] == band]
            if len(cell_candidates) < args.per_cell:
                shortfalls.append((cluster_id, band, len(cell_candidates)))
            selected, reasons = select_cell(cell_candidates, args.per_cell, rng)
            for rank, (c, reason) in enumerate(zip(selected, reasons), start=1):
                final_records.append({
                    "image": c["image"], "filename": c["filename"], "prefix": c["prefix"],
                    "cluster_id": c["cluster_id"], "centrality": c["band"],
                    "distance_to_centroid": c["distance_to_centroid"],
                    "rank_within_cluster": c["rank_within_cluster"],
                    "percentile_within_cluster": c["percentile_within_cluster"],
                    "question_levels": sorted(c["levels"]), "features": sorted(c["features"]),
                    "n_questions": c["n_questions"],
                    "selection_order_in_cell": rank, "selection_reason": reason,
                })
            print(f"cluster {cluster_id} / {band}: selected {len(selected)}/{args.per_cell} "
                  f"from {len(cell_candidates)} candidates")

    if shortfalls:
        print(f"\nWARNING -- {len(shortfalls)} cell(s) had fewer than {args.per_cell} eligible candidates:")
        for cid, band, n in shortfalls:
            print(f"  cluster {cid} / {band}: only {n} available")

    assert len(final_records) == len({r["image"] for r in final_records}), "duplicate images in final selection!"
    assert not (set(r["filename"] for r in final_records) & excluded), "overlap with excluded 100-subset!"
    print(f"\nFinal selection: {len(final_records)} images, no duplicates, no overlap with prior 100-subset (verified)")

    with open(f"{args.output_prefix}.json", "w") as f:
        json.dump(final_records, f, indent=2)
    print(f"Saved -> {args.output_prefix}.json")

    with open(f"{args.output_prefix}_selection_report.csv", "w", newline="") as f:
        fieldnames = ["image", "filename", "prefix", "cluster_id", "centrality", "distance_to_centroid",
                      "rank_within_cluster", "percentile_within_cluster", "question_levels", "features",
                      "n_questions", "selection_order_in_cell", "selection_reason"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in final_records:
            row = dict(r)
            row["question_levels"] = ";".join(row["question_levels"])
            row["features"] = ";".join(row["features"])
            w.writerow(row)
    print(f"Saved -> {args.output_prefix}_selection_report.csv")

    dist = {
        "by_cluster": {c: sum(1 for r in final_records if r["cluster_id"] == c) for c in range(K)},
        "by_centrality": {b: sum(1 for r in final_records if r["centrality"] == b) for b in BANDS},
        "by_question_level": {},
        "by_feature_top20": {},
        "shortfall_cells": [{"cluster_id": cid, "centrality": band, "n_available": n} for cid, band, n in shortfalls],
    }
    level_counts = defaultdict(int)
    feature_counts = defaultdict(int)
    for r in final_records:
        for lv in r["question_levels"]:
            level_counts[lv] += 1
        for ft in r["features"]:
            feature_counts[ft] += 1
    dist["by_question_level"] = dict(sorted(level_counts.items(), key=lambda kv: -kv[1]))
    dist["by_feature_top20"] = dict(sorted(feature_counts.items(), key=lambda kv: -kv[1])[:20])
    dist["n_unique_features_covered"] = len(feature_counts)
    dist["n_unique_levels_covered"] = len(level_counts)

    with open(f"{args.output_prefix}_distribution.json", "w") as f:
        json.dump(dist, f, indent=2)
    print(f"Saved -> {args.output_prefix}_distribution.json")

    print("\n=== Final distribution ===")
    print("by cluster:", dist["by_cluster"])
    print("by centrality:", dist["by_centrality"])
    print("by question level:", dist["by_question_level"])
    print(f"unique features covered: {dist['n_unique_features_covered']}")
    print("top 20 features:", dist["by_feature_top20"])

    print("\n\n=== Generating comprehensive coverage report (additive -- selection above is unchanged) ===")
    generate_coverage_report(final_records, pool, args.output_prefix)

if __name__ == "__main__":
    main()
