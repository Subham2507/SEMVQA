from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    ChatSFTDataset,
    DataCollatorPad,
    check_transformers,
    load_model,
    load_processor,
    prepare_for_generation,
    read_jsonl,
    stage2_build_fn,
    stage2_evidence_build_fn,
)
from train_stage2_vqa import SYSTEM_PROMPT, SYSTEM_PROMPT_EVIDENCE

def _norm(s: str) -> list[str]:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.split()

def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)

def rouge_l(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return float(p == g)
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == g[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(p)][len(g)]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="train/runs/stage2/checkpoint-22500")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--split", default="test", choices=["test", "val", "train", "benchmark"])
    p.add_argument("--out-dir", default=None, help="default: <adapter-run>/eval")

    p.add_argument("--loss-batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--loss-n", type=int, default=0, help="cap rows for the loss pass (0 = all)")

    p.add_argument("--gen-n", type=int, default=500, help="rows to generate on (0 = skip, -1 = all)")
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--min-pixels", type=int, default=200704)
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--no-system-prompt", action="store_true")

    p.add_argument("--target-format", choices=["answer", "evidence_answer"], default="answer",
                   help="must match the adapter's training --target-format")
    p.add_argument("--gpu-mem-fraction", type=float, default=0.85,
                   help="hard-cap VRAM use (0.85 ~= 42 GB of 49); 0 disables")
    return p.parse_args()

def _format_pair(args):
    if getattr(args, "target_format", "answer") == "evidence_answer":
        return stage2_evidence_build_fn, SYSTEM_PROMPT_EVIDENCE
    return stage2_build_fn, SYSTEM_PROMPT

@torch.inference_mode()
def loss_pass(model, processor, rows, args) -> dict:

    build_fn, sys_default = _format_pair(args)
    ds = ChatSFTDataset(rows, processor, build_fn, args.max_length,
                        None if args.no_system_prompt else sys_default)
    dl = DataLoader(ds, batch_size=args.loss_batch_size, shuffle=False, num_workers=4,
                    collate_fn=DataCollatorPad(pad_token_id=processor.tokenizer.pad_token_id))
    tot_loss, tot_tok = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(dl):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        n_tok = int((batch["labels"] != -100).sum())
        tot_loss += float(out.loss) * n_tok
        tot_tok += n_tok
        if i % 50 == 0:
            print(f"  loss pass {i * args.loss_batch_size}/{len(ds)}  "
                  f"({(time.time() - t0):.0f}s)", flush=True)
    mean = tot_loss / max(tot_tok, 1)
    return {"n_rows": len(ds), "loss": mean, "perplexity": float(torch.tensor(mean).exp()),
            "answer_tokens": tot_tok}

@torch.inference_mode()
def gen_pass(model, processor, rows, args) -> tuple[list[dict], dict]:
    tok = processor.tokenizer
    tok.padding_side = "left"

    sys_prompt = None if args.no_system_prompt else _format_pair(args)[1]
    preds: list[dict] = []
    t0 = time.time()

    for start in range(0, len(rows), args.gen_batch_size):
        chunk = rows[start : start + args.gen_batch_size]
        from PIL import Image

        images, texts = [], []
        for r in chunk:
            with Image.open(r["image_path"]) as im:
                images.append(im.convert("RGB"))
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"]}]}
            ]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1] :]
        for r, g in zip(chunk, gen):
            pred_raw = tok.decode(g, skip_special_tokens=True).strip()

            pred = pred_raw
            if args.target_format == "evidence_answer" and "Answer:" in pred_raw:
                pred = pred_raw.split("Answer:", 1)[1].strip()
            preds.append({"image_name": r["image_name"], "level": r.get("level"),
                          "question": r["question"], "gold": r["answer"], "pred": pred,
                          "pred_full": pred_raw})
        if start % (args.gen_batch_size * 10) == 0:
            print(f"  gen {start}/{len(rows)}  ({(time.time() - t0):.0f}s)", flush=True)

    by_level: dict[str, list] = defaultdict(list)
    for p in preds:
        f1, rl = token_f1(p["pred"], p["gold"]), rouge_l(p["pred"], p["gold"])
        p["token_f1"], p["rouge_l"] = round(f1, 4), round(rl, 4)
        by_level["_all"].append(p)
        by_level[p["level"] or "none"].append(p)

    def agg(items):
        return {
            "n": len(items),
            "token_f1": round(sum(x["token_f1"] for x in items) / len(items), 4),
            "rouge_l": round(sum(x["rouge_l"] for x in items) / len(items), 4),
            "avg_pred_words": round(sum(len(x["pred"].split()) for x in items) / len(items), 1),
            "avg_gold_words": round(sum(len(x["gold"].split()) for x in items) / len(items), 1),
        }

    metrics = {lvl: agg(items) for lvl, items in sorted(by_level.items())}
    return preds, metrics

def main() -> None:
    args = parse_args()
    check_transformers()

    from common import cap_gpu_memory

    cap_gpu_memory(args.gpu_mem_fraction)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.adapter.rstrip("/")) if "checkpoint" in args.adapter else args.adapter, "eval"
    )
    os.makedirs(out_dir, exist_ok=True)

    rows = read_jsonl(os.path.join(args.data_dir, f"stage2_{args.split}.jsonl"))
    print(f"{args.split} split: {len(rows)} rows")

    processor = load_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_model(args.model, dtype=args.dtype, attn_implementation=args.attn)
    if args.adapter and args.adapter.lower() != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        print(f"loaded adapter: {args.adapter}")
    model.eval().cuda()
    prepare_for_generation(model, processor)

    report: dict = {"adapter": args.adapter, "split": args.split, "n_rows": len(rows),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    loss_rows = rows[: args.loss_n] if args.loss_n else rows
    print(f"\n[1/2] teacher-forced loss over {len(loss_rows)} rows ...")
    report["loss"] = loss_pass(model, processor, loss_rows, args)
    print(f"  -> loss {report['loss']['loss']:.4f}  ppl {report['loss']['perplexity']:.3f}")

    if args.gen_n != 0:
        n = len(rows) if args.gen_n < 0 else min(args.gen_n, len(rows))
        import random

        gen_rows = random.Random(args.seed).sample(rows, n) if n < len(rows) else rows
        print(f"\n[2/2] generating on {n} rows (bs {args.gen_batch_size}) ...")
        preds, metrics = gen_pass(model, processor, gen_rows, args)
        report["generation"] = metrics

        pred_path = os.path.join(out_dir, f"{args.split}_predictions.jsonl")
        with open(pred_path, "w") as fh:
            for p in preds:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

        import csv

        csv_path = os.path.join(out_dir, f"{args.split}_predictions.csv")
        cols = ["image_name", "level", "token_f1", "rouge_l", "question", "gold", "pred"]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(preds)
        print(f"  predictions -> {pred_path}\n               {csv_path}")
        print(f"  overall: token_f1 {metrics['_all']['token_f1']}  rouge_l {metrics['_all']['rouge_l']}")
        for lvl, m in metrics.items():
            if lvl not in ("_all",):
                print(f"    {lvl:16} n={m['n']:<5} f1={m['token_f1']}  rougeL={m['rouge_l']}")

    metrics_path = os.path.join(out_dir, f"{args.split}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nmetrics -> {metrics_path}")

if __name__ == "__main__":
    main()
