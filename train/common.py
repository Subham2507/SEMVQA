from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset

def cap_gpu_memory(fraction: float, device: int = 0) -> None:
    if fraction and 0 < fraction < 1 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"GPU memory capped at {fraction:.0%}  (~{total * fraction:.1f} GB of {total:.1f} GB)")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "Qwen3.5-2B")
DEFAULT_IMAGES = os.path.join(PROJECT_ROOT, "need_images")
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "need_text.csv")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

IM_END = "<|im_end|>\n"

SUMMARY_PROMPTS = [
    "Provide a detailed scientific description of this scanning electron micrograph of a "
    "materials-science specimen.",
    "You are a materials scientist. Summarize the microstructural features visible in this SEM image.",
    "Describe this scanning electron micrograph, focusing on morphology, phases and any notable features.",
    "Write a concise technical summary of what this SEM image shows about the material's microstructure.",
    "Analyze this scanning electron micrograph and describe the key structural characteristics of the sample.",
]

def figure_id(image_name: str) -> str:
    base = os.path.splitext(os.path.basename(image_name))[0]
    return base.rsplit("_", 1)[0] if "_" in base else base

def pick_prompt(key: str, mode: str = "rotate") -> str:
    if mode == "fixed":
        return SUMMARY_PROMPTS[0]
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return SUMMARY_PROMPTS[h % len(SUMMARY_PROMPTS)]

def check_transformers() -> None:
    import transformers
    from packaging import version

    if version.parse(transformers.__version__.split("+")[0]) < version.parse("4.57.0"):
        raise RuntimeError(
            f"transformers {transformers.__version__} is too old for the qwen3_5 architecture.\n"
            f"    pip install -U 'transformers>=4.57.0' accelerate\n"
            f"(the model config declares model_type='qwen3_5', added in transformers 4.57)."
        )

def load_processor(model_path: str, min_pixels: int, max_pixels: int):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    try:
        processor.image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    except Exception:
        pass
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = "<|endoftext|>"
    return processor

def prepare_for_generation(model, processor):
    tok = processor.tokenizer
    model.config.use_cache = True
    gc = model.generation_config
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    gc.eos_token_id = [i for i in {im_end, tok.eos_token_id, tok.pad_token_id} if i is not None]
    gc.pad_token_id = tok.pad_token_id
    return model

def load_model(model_path: str, dtype: str = "bfloat16", attn_implementation: str = "sdpa"):
    from transformers import AutoModelForImageTextToText

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model

_VISION_HINTS = ("visual", "vision", "patch_embed", "merger", "deepstack", "image_encoder")
_SKIP_HINTS = ("lm_head", "embed_tokens", "embed_", ".mtp", "mtp.")

def find_lora_targets(model, freeze_vision: bool = True) -> list[str]:
    import torch.nn as nn

    names: list[str] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        low = full_name.lower()
        if any(h in low for h in _SKIP_HINTS):
            continue
        if freeze_vision and any(h in low for h in _VISION_HINTS):
            continue
        names.append(full_name)
    return sorted(names)

def summarize_targets(targets: list[str]) -> str:
    import collections

    leaves = collections.Counter(t.split(".")[-1] for t in targets)
    n_vis = sum(1 for t in targets if any(h in t.lower() for h in _VISION_HINTS))
    return (f"{len(targets)} Linear modules ({len(targets) - n_vis} text, {n_vis} vision) | "
            + ", ".join(f"{k}:{v}" for k, v in sorted(leaves.items())))

def apply_lora(model, r: int, alpha: int, dropout: float, freeze_vision: bool,
               full_vision: bool = False, target_modules: list[str] | None = None):
    from peft import LoraConfig, get_peft_model

    if full_vision:
        freeze_vision = True
    targets = target_modules or find_lora_targets(model, freeze_vision=freeze_vision)
    if not targets:
        raise RuntimeError("No LoRA target modules found - inspect model.named_modules().")

    modules_to_save = ["visual"] if full_vision else None
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()
    return model, targets

def load_adapter_for_training(model, adapter_path: str, merge: bool, freeze_vision: bool = True):
    from peft import PeftModel

    if merge:
        m = PeftModel.from_pretrained(model, adapter_path)
        m = m.merge_and_unload()
        return m, None

    m = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    m.enable_input_require_grads()

    frozen = 0
    if freeze_vision:
        for name, p in m.named_parameters():
            if p.requires_grad and any(h in name.lower() for h in _VISION_HINTS):
                p.requires_grad = False
                frozen += 1
    if frozen:
        print(f"froze {frozen} vision LoRA tensors carried over from the stage-1 adapter")

    all_targets = list(getattr(m.peft_config["default"], "target_modules", []) or [])
    if freeze_vision:
        all_targets = [t for t in all_targets if not any(h in t.lower() for h in _VISION_HINTS)]
    return m, all_targets

@dataclass
class ChatSFTDataset(Dataset):

    records: list[dict]
    processor: Any
    build_fn: Any
    max_length: int = 1536
    system_prompt: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _messages(self, instruction: str) -> list[dict]:
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": instruction}]})
        return msgs

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        rec = self.records[idx]
        instruction, target = self.build_fn(rec)
        instruction, target = instruction.strip(), target.strip()

        with Image.open(rec["image_path"]) as im:
            image = im.convert("RGB")

        prompt_text = self.processor.apply_chat_template(
            self._messages(instruction), tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target + IM_END

        full = self.processor(text=[full_text], images=[image], return_tensors="pt", padding=False)
        prompt = self.processor(text=[prompt_text], images=[image], return_tensors="pt", padding=False)

        input_ids = full["input_ids"][0]
        prompt_len = int(prompt["input_ids"].shape[1])

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        aligned = {"input_ids": input_ids, "labels": labels}
        for k in ("mm_token_type_ids", "token_type_ids"):
            if k in full:
                aligned[k] = full[k][0]

        keep = min(input_ids.shape[0], self.max_length)
        item = {k: v[:keep] for k, v in aligned.items()}
        item["attention_mask"] = torch.ones(keep, dtype=torch.long)
        item["pixel_values"] = full["pixel_values"]
        for k in ("image_grid_thw", "image_sizes"):
            if k in full:
                item[k] = full[k]
        return item

@dataclass
class DataCollatorPad:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].size(0) for f in features)

        def pad(seq: torch.Tensor, value: int) -> torch.Tensor:
            if seq.size(0) == max_len:
                return seq
            return torch.cat([seq, seq.new_full((max_len - seq.size(0),), value)])

        input_ids = torch.stack([pad(f["input_ids"], self.pad_token_id) for f in features])
        labels = torch.stack([pad(f["labels"], -100) for f in features])
        attention_mask = torch.zeros_like(input_ids)
        for i, f in enumerate(features):
            attention_mask[i, : f["input_ids"].size(0)] = 1

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
        }
        for k in ("mm_token_type_ids", "token_type_ids"):
            if k in features[0]:
                batch[k] = torch.stack([pad(f[k], 0) for f in features])
        for k in ("image_grid_thw", "image_sizes"):
            if k in features[0]:
                batch[k] = torch.cat([f[k] for f in features], dim=0)
        return batch

DataCollatorStage1 = DataCollatorPad

def stage1_build_fn(prompt_mode: str = "rotate"):
    return lambda rec: (pick_prompt(rec["figure_id"], prompt_mode), rec["summary"])

def stage2_build_fn(rec: dict) -> tuple[str, str]:
    return rec["question"], rec["answer"]

def stage2_evidence_build_fn(rec: dict) -> tuple[str, str]:
    return rec["question"], (
        f"Evidence: {(rec.get('visual_evidence') or '').strip()}\n"
        f"Answer: {(rec.get('answer') or '').strip()}"
    )

def make_jsonl_logger_callback(path: str):
    import json as _json
    import time

    from transformers import TrainerCallback

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    class JsonlLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.is_world_process_zero:
                return
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "step": state.global_step,
                   "epoch": round(state.epoch, 4) if state.epoch is not None else None,
                   **logs}
            with open(path, "a") as fh:
                fh.write(_json.dumps(row) + "\n")

    return JsonlLogger()

def read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]

def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
