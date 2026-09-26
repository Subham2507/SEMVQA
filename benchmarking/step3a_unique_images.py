import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

VQA = "."
PREFIXES = ("ceramics_prod_", "composite_", "ni_alloy_")

def prefix_of(fn):
    for p in PREFIXES:
        if fn.startswith(p):
            return p
    return "img"

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default=f"{VQA}/images_full/images_full")

    ap.add_argument("--corpus", default=f"{VQA}/sem_vqa_corpus.jsonl")
    ap.add_argument("--output-dir", default=f"{VQA}/training_data")
    args = ap.parse_args()

    train, test = set(), set()
    for line in open(args.corpus):
        if not line.strip():
            continue
        r = json.loads(line)
        (train if r.get("split") == "train" else test if r.get("split") == "test" else set()).add(r["filename"])

    files = sorted(os.listdir(args.image_dir))
    print(f"hashing {len(files)} files ...")
    by_content = defaultdict(list)
    for i, fn in enumerate(files, 1):
        p = os.path.join(args.image_dir, fn)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        by_content[(sz, md5(p))].append(fn)
        if i % 20000 == 0:
            print(f"  {i}/{len(files)}")
    print(f"{len(by_content)} distinct image contents")

    def rows_for(restrict=None):
        out = []
        for (sz, h), fns in by_content.items():
            aliases = sorted(fns) if restrict is None else sorted(f for f in fns if f in restrict)
            if not aliases:
                continue
            in_tr = sorted(f for f in aliases if f in train)
            in_te = sorted(f for f in aliases if f in test)
            out.append({
                "md5": h, "size_bytes": sz, "canonical_filename": aliases[0],
                "n_copies": len(aliases), "aliases": ";".join(aliases),
                "splits": ";".join(sorted({("train" if f in train else "test" if f in test
                                             else "neither") for f in aliases})),
                "in_train": len(in_tr), "in_test": len(in_te),
                "split_conflict": int(bool(in_tr) and bool(in_te)),
                "prefixes": ";".join(sorted({prefix_of(f) for f in aliases})),
            })
        out.sort(key=lambda r: (-r["n_copies"], r["canonical_filename"]))
        return out

    all_rows = rows_for(None)
    k85_rows = rows_for(train | test)

    for name, rows in [("unique_images.csv", all_rows), ("unique_images_corpus.csv", k85_rows)]:
        path = os.path.join(args.output_dir, name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {path}")

    def stats(rows, label, n_files):
        conflicts = sum(r["split_conflict"] for r in rows)
        multi = sum(1 for r in rows if r["n_copies"] > 1)
        tr_only = sum(1 for r in rows if r["in_train"] and not r["in_test"])
        te_only = sum(1 for r in rows if r["in_test"] and not r["in_train"])
        print(f"\n{label}: {n_files} files -> {len(rows)} unique images "
              f"({n_files - len(rows)} redundant copies)")
        print(f"  contents with >1 copy         : {multi}")
        print(f"  train-only contents           : {tr_only}")
        print(f"  test-only contents (truly held out): {te_only}")
        print(f"  SPLIT CONFLICTS (in both)     : {conflicts}")
        return {"n_files": n_files, "n_unique": len(rows), "redundant_copies": n_files - len(rows),
                "contents_with_multiple_copies": multi, "train_only": tr_only,
                "test_only_truly_held_out": te_only, "split_conflicts": conflicts}

    s_all = stats(all_rows, "ALL files on disk", len(files))
    s_k85 = stats(k85_rows, "corpus (train+test)", len(train | test))

    summary = {"all_on_disk": s_all, "corpus": s_k85,
                "canonical_choice": "alphabetically-first filename in each content group",
                "note": "split_conflict=1 marks contents whose copies straddle train and test "
                        "-- a data-leakage risk for any model evaluated on the test split"}
    sp = os.path.join(args.output_dir, "unique_images_summary.json")
    json.dump(summary, open(sp, "w"), indent=2)
    print(f"\nWrote {sp}")

if __name__ == "__main__":
    main()
