import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def unique_images(eval_files, image_dir="images"):
    paths = set()
    for fn in eval_files:
        with open(fn) as f:
            if fn.endswith(".jsonl"):
                records = (json.loads(line) for line in f if line.strip())
            else:
                records = json.load(f)
            for r in records:
                if "image" in r:
                    paths.add(r["image"])
                else:
                    paths.add(os.path.join(image_dir, r["filename"]))
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-files", nargs="+", default=["sem_vqa_corpus.jsonl"])
    ap.add_argument("--image-dir", default="images",
                    help="prepended to bare filenames taken from the corpus JSONL")
    ap.add_argument("--model-id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--output-dir", default="training_data/dinov3_clusters")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                     help="checkpoint to disk every N images")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    paths_path = os.path.join(args.output_dir, "image_paths.json")

    all_images = unique_images(args.eval_files, args.image_dir)
    print(f"{len(all_images)} unique images across {args.eval_files}")

    done_paths = []
    done_embs = None
    if os.path.exists(emb_path) and os.path.exists(paths_path):
        done_embs = np.load(emb_path)
        with open(paths_path) as f:
            done_paths = json.load(f)
        print(f"Resuming: {len(done_paths)} images already embedded, skipping those.")

    done_set = set(done_paths)
    remaining = [p for p in all_images if p not in done_set]
    print(f"{len(remaining)} images remaining")

    if not remaining:
        print("Nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device).eval()

    new_embs = []
    new_paths = []

    def checkpoint():
        embs = np.concatenate([done_embs, np.stack(new_embs)], axis=0) if done_embs is not None and new_embs else               (np.stack(new_embs) if new_embs else done_embs)
        paths = done_paths + new_paths
        np.save(emb_path, embs)
        with open(paths_path, "w") as f:
            json.dump(paths, f)
        print(f"Checkpointed {len(paths)}/{len(all_images)} -> {args.output_dir}")

    since_checkpoint = 0
    for i in range(0, len(remaining), args.batch_size):
        batch_paths = remaining[i:i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs)
        cls_emb = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        new_embs.extend(list(cls_emb))
        new_paths.extend(batch_paths)
        since_checkpoint += len(batch_paths)
        print(f"{len(done_paths) + len(new_paths)}/{len(all_images)}")
        if since_checkpoint >= args.checkpoint_every:
            checkpoint()
            since_checkpoint = 0

    checkpoint()
    print("Done.")

if __name__ == "__main__":
    main()
