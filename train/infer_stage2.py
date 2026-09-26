from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    check_transformers,
    load_model,
    load_processor,
    prepare_for_generation,
    read_jsonl,
)
from train_stage2_vqa import SYSTEM_PROMPT, SYSTEM_PROMPT_EVIDENCE

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--image", default=None)
    p.add_argument("--question", default=None)
    p.add_argument("--split", default="test", choices=["test", "val", "train", "benchmark"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--system-prompt", default=SYSTEM_PROMPT,
                   help="explicit override; if left at the default it follows --target-format")

    p.add_argument("--target-format", choices=["answer", "evidence_answer"], default="answer")
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--min-pixels", type=int, default=200704)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--max-new-tokens", type=int, default=200)
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

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval().cuda()
    prepare_for_generation(model, processor)

    if args.image and args.question:
        items = [{"image_path": args.image, "question": args.question, "answer": None}]
    else:
        items = read_jsonl(os.path.join(args.data_dir, f"stage2_{args.split}.jsonl"))[: args.n]

    _default_sys = (SYSTEM_PROMPT_EVIDENCE if args.target_format == "evidence_answer"
                    else SYSTEM_PROMPT)
    _sys_text = args.system_prompt if args.system_prompt != SYSTEM_PROMPT else _default_sys
    sys_prompt = None if args.no_system_prompt else _sys_text
    for it in items:
        with Image.open(it["image_path"]) as im:
            image = im.convert("RGB")
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": it["question"]}]})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        print("=" * 100)
        print("image :", os.path.basename(it["image_path"]))
        print("Q     :", it["question"])
        print("pred  :", pred)
        if it.get("answer"):
            print("gold  :", it["answer"])

if __name__ == "__main__":
    main()
