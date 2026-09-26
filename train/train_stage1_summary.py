from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    ChatSFTDataset,
    DataCollatorPad,
    apply_lora,
    check_transformers,
    load_model,
    load_processor,
    make_jsonl_logger_callback,
    read_jsonl,
    stage1_build_fn,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "stage1"))
    p.add_argument("--train-file", default=None, help="override (default: <data-dir>/stage1_train.jsonl)")
    p.add_argument("--val-file", default=None)
    p.add_argument("--max-train", type=int, default=0, help="debug: cap training rows")
    p.add_argument("--max-val", type=int, default=512, help="cap eval rows for speed (0 = all)")

    p.add_argument("--min-pixels", type=int, default=200704, help="448*448")
    p.add_argument("--max-pixels", type=int, default=1048576, help="1024*1024")
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--prompt-mode", choices=["rotate", "fixed"], default="rotate")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--train-vision", action="store_true", help="also add LoRA to the vision tower")
    p.add_argument("--full-vision", action="store_true",
                   help="fully fine-tune + save the whole vision tower (text side stays LoRA); "
                        "heavier, use for strong domain shift. Overrides --train-vision.")
    p.add_argument("--list-targets", action="store_true", help="print chosen LoRA modules and exit")

    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmup-steps", type=int, default=0, help="overrides --warmup-ratio when > 0")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--no-grad-checkpointing", action="store_true")

    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gpu-mem-fraction", type=float, default=0.85,
                   help="hard-cap VRAM use (0.85 ~= 42 GB of 49); 0 disables the cap")
    p.add_argument("--report-to", default="none",
                   help="none | tensorboard | wandb (install the backend yourself)")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    check_transformers()

    import transformers
    from transformers import Trainer, TrainingArguments

    from common import cap_gpu_memory

    cap_gpu_memory(args.gpu_mem_fraction)

    train_file = args.train_file or os.path.join(args.data_dir, "stage1_train.jsonl")
    val_file = args.val_file or os.path.join(args.data_dir, "stage1_val.jsonl")
    for f in (train_file, val_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found - run:  python train/build_splits.py")

    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    if args.max_train and args.max_train < len(train_rows):
        import random

        train_rows = random.Random(args.seed).sample(train_rows, args.max_train)
    if args.max_val and args.max_val < len(val_rows):
        import random

        val_rows = random.Random(args.seed).sample(val_rows, args.max_val)
    print(f"train rows: {len(train_rows)} | val rows: {len(val_rows)}")

    print("loading processor + model ...")
    processor = load_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_model(args.model, dtype=args.dtype, attn_implementation=args.attn)

    model, targets = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        freeze_vision=not (args.train_vision or args.full_vision),
        full_vision=args.full_vision,
    )
    from common import summarize_targets

    vision_mode = "full fine-tune" if args.full_vision else ("LoRA" if args.train_vision else "frozen")
    print(f"vision tower: {vision_mode}")
    print(f"LoRA targets: {summarize_targets(targets)}")
    model.print_trainable_parameters()
    if args.list_targets:
        return

    build_fn = stage1_build_fn(args.prompt_mode)
    train_ds = ChatSFTDataset(train_rows, processor, build_fn, args.max_length)
    val_ds = ChatSFTDataset(val_rows, processor, build_fn, args.max_length)
    collator = DataCollatorPad(pad_token_id=processor.tokenizer.pad_token_id)

    import math

    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum * world))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = args.warmup_steps if args.warmup_steps else int(args.warmup_ratio * total_steps)
    print(f"~{total_steps} optimizer steps | warmup {warmup_steps}")

    import json
    import platform
    import time

    import peft as _peft
    import torch as _torch

    os.makedirs(args.output_dir, exist_ok=True)
    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "resolved": {
            "train_file": train_file,
            "val_file": val_file,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "world_size": world,
            "effective_batch_size": args.batch_size * args.grad_accum * world,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "warmup_steps": warmup_steps,
            "vision_mode": vision_mode,
            "gradient_checkpointing": not args.no_grad_checkpointing,
        },
        "lora_target_modules": targets,
        "versions": {
            "python": platform.python_version(),
            "torch": _torch.__version__,
            "transformers": transformers.__version__,
            "peft": _peft.__version__,
            "cuda": _torch.version.cuda,
            "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
        },
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2, default=str)
    print(f"run config          -> {os.path.join(args.output_dir, 'run_config.json')}")

    use_gc = not args.no_grad_checkpointing
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=args.report_to,
        seed=args.seed,
    )

    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        callbacks=[make_jsonl_logger_callback(log_path)],
    )

    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    import json

    with open(os.path.join(args.output_dir, "log_history.json"), "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)

    print(f"\nsaved LoRA adapter -> {final_dir}")
    print(f"training log        -> {log_path}  (+ log_history.json)")
    print("merge for inference with:  python train/infer_stage1.py --adapter", final_dir)

if __name__ == "__main__":
    main()
