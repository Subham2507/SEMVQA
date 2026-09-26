import json, os, statistics

OUT = "."
FMT_A = "./eval_results_evidence"
RUNS = ["stage2_from_base", "stage2_from_frozen", "stage2_from_train_vision_evi"]

print(f"{'adapter':32s} {'n':>5} {'Evid':>5} {'Ans:':>5} {'words':>6} {'lossB':>7}")
print("(lossB = loss vs Format B targets. NOT comparable to the Format A loss in")
print(" eval_results_evidence/ -- different target token distribution. Reported for")
print(" the record only; the compliance count is the actual result.)")
tot = comp = 0
for r in RUNS:
    p = f"{OUT}/{r}/benchmark_predictions.jsonl"
    if not os.path.exists(p):
        print(f"{r:32s}  (not run yet)"); continue
    rows = [json.loads(l) for l in open(p)]
    k = 'pred_full' if 'pred_full' in rows[0] else 'pred'
    ev = sum(1 for x in rows if (x.get(k) or '').lstrip().startswith('Evidence:'))
    an = sum(1 for x in rows if 'Answer:' in (x.get(k) or ''))
    w = statistics.mean(len((x.get(k) or '').split()) for x in rows)
    lb = json.load(open(f"{OUT}/{r}/benchmark_metrics.json"))['loss']['loss']
    print(f"{r:32s} {len(rows):5d} {ev:5d} {an:5d} {w:6.1f} {lb:7.4f}")
    tot += len(rows); comp += ev
if tot:
    print(f"\nTOTAL evidence compliance: {comp}/{tot} ({100*comp/tot:.1f}%)")
    print("Reference points: prompting-only n=24 -> 0/24 ; format-trained -> 39144/39144 (100%)")
    print("NB: token_f1 here is scored against Evidence:+Answer: gold while the model emits a")
    print("    bare answer -- it measures format mismatch, NOT answer quality. Do not report it.")
