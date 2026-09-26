from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    SUMMARY_PROMPTS,
    check_transformers,
    load_model,
    load_processor,
    prepare_for_generation,
    read_jsonl,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (omit to run the base model)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--image", default=None, help="single image path (overrides --n test sampling)")
    p.add_argument("--n", type=int, default=5, help="number of test images to sample")
    p.add_argument("--prompt", default=SUMMARY_PROMPTS[0])
    p.add_argument("--min-pixels", type=int, default=200704)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return p.parse_args()

def main() -> None:
    args = parse_args()
    check_transformers()
    from PIL import Image

    processor = load_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_model(args.model, dtype=args.dtype, attn_implementation=args.attn)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval().cuda()
    prepare_for_generation(model, processor)

    if args.image:
        items = [{"image_path": args.image, "summary": None}]
    else:
        rows = read_jsonl(os.path.join(args.data_dir, "stage1_test.jsonl"))
        items = rows[: args.n]

    for it in items:
        with Image.open(it["image_path"]) as im:
            image = im.convert("RGB")
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        pred = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

        print("=" * 100)
        print("image :", os.path.basename(it["image_path"]))
        print("pred  :", pred)
        if it.get("summary"):
            print("gold  :", it["summary"])

if __name__ == "__main__":
    main()
