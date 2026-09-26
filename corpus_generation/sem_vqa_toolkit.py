# -*- coding: utf-8 -*-
"""
SEM-VQA toolkit -- the two pipelines used to build and evaluate this project's SEM-micrograph
VQA corpus, packaged together as one file with two subcommands.

  generate   QA-pair generation over SEM micrographs. Prompt version v7_four_level_grounded,
             gemini-2.5-flash via Vertex AI. This is the exact prompt and pipeline logic used
             to produce the released corpus (309,518 QA pairs over 66,385 SEM micrographs).

  answer     Benchmark answering: runs a model over a frozen set of (image, question) records
             and writes predictions in the schema used for scoring. This is the Gemini arm
             (gemini-3.1-flash-lite, Vertex AI) of the rep300 answering benchmark. Two other
             proprietary arms (gpt-5-mini via Azure OpenAI, claude-haiku-4-5 via Azure
             Anthropic) were run against the identical record schema but a different SDK each;
             they are not included here so this file's only dependency is `google-genai`. See
             the README for the exact record schema if you want to add another provider arm.

No API key is read or stored anywhere in this file. Both subcommands authenticate to Vertex AI
via Application Default Credentials (`gcloud auth application-default login`), which is the
credential path this project actually used -- there is no key to redact because none was used.

Install:
    pip install google-genai

Usage:
    python corpus_generation/sem_vqa_toolkit.py generate --index index.json --images-dir ./images \
        --out sem_vqa.jsonl --project YOUR_GCP_PROJECT --n 100

    python corpus_generation/sem_vqa_toolkit.py answer --benchmark benchmark.json --images-dir ./images \
        --out predictions.jsonl --format evidence --project YOUR_GCP_PROJECT
"""
import argparse
import base64
import json
import logging
import random
import re
import time
from pathlib import Path

logger = logging.getLogger("sem_vqa_toolkit")
logger.setLevel(logging.INFO)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_console)


# ============================================================================================
# SUBCOMMAND 1: generate
#
# Adapted from generate_vqa_v7.py, the script that produced the released corpus. The prompt
# text below (PROMPT_TEMPLATE), the four-level taxonomy, the anti-hallucination rules, and the
# stratified-by-source-prefix sampling are unchanged from that run. What changed for this
# release: hardcoded absolute paths and the GCP project id were replaced with CLI arguments,
# since those are specific to the machine the corpus was originally generated on, not part of
# the method.
# ============================================================================================

GENERATE_PROMPT_VERSION = "v7_four_level_grounded"
GENERATE_MAX_QUESTIONS = 5
VALID_LEVELS = {"observation", "detection", "identification", "interpretation"}

PROMPT_TEMPLATE = """You are an expert Metallurgist and Materials Scientist annotating VQA
training data that will teach a vision-language model to read SEM microstructure the way you do.
The model will only ever see the image at inference -- never this caption or summary -- so every
question must be answerable from the image alone. It's fine if your answer agrees with the
caption/summary (they describe what's visible, so agreement is a grounding signal) -- what's NOT
allowed is phrasing a question so it can be answered by quoting or rephrasing the caption/summary
text without looking at the image.

## GOAL
Teach the model to recognize and reason about the microstructural features actually present in
this image. Prefer scientifically meaningful questions (about real microstructural features --
cracks, pores, grains, boundaries, particles, precipitates, fracture morphology, phases, surface
texture) over superficial administrative questions (is a scale bar present? is there text?).
Generate only questions that are strongly supported by the image -- if an image genuinely
supports only one simple question, generate only that one.

## HOW MANY QUESTIONS
Decide how many this image genuinely supports, up to 5. There is no minimum. Do NOT pad. The
questions on one image should VARY by covering DIFFERENT features or DIFFERENT aspects of the
microstructure (e.g. one on grain morphology, one on a visible defect, one on phase contrast) --
not the same feature re-asked in different words. Two questions that teach the same visual fact
are one question; drop the weaker.

## THE FOUR LEVELS (use whichever the image supports; they build on each other)
1. OBSERVATION -- a broad "what do you see / describe the microstructure" summary. At most one.
   A good observation answer summarizes the most salient VISIBLE microstructural features in
   decreasing order of importance. Mention only features that are actually visible. Do NOT pad
   the answer with generic statements like "this is an SEM image" or "the image shows a
   microstructure."
2. DETECTION -- identify what features are present, what they look like, and where they are.
   PREFER open, discovery-style questions that make the answerer RECOGNIZE the feature rather than
   confirm a named one:
     - good: "What defects or discontinuities are visible in the microstructure?"
     - good: "What features are present along the grain boundaries?"
     - good (detail of a visibly present feature): "What is the morphology of the particles?",
       "Which region contains the most porosity?", "Which area shows rougher surface texture?"
     - avoid: "Are there pores?", "Is a precipitate visible?" -- these name the answer inside the
       question, so the model only has to confirm it, not recognize it.
   A closed "Is X present?" question is allowed ONLY when the answer is genuinely in doubt (the
   feature could plausibly be absent) -- never ask it about a feature that obviously dominates the
   image, since that makes the question rhetorical and every such answer is "yes."
   Localization (where / which region) is an important microscopy skill -- use it where the image
   supports it.
3. IDENTIFICATION -- naming what a feature IS (e.g. "the dark network is a carbide network",
   "this is the alpha phase"). Only name a technical constituent if the caption/summary
   corroborates it (see grounding rule). If the caption/summary does NOT confirm the technical
   term, describe the feature by its appearance only (shape, contrast, arrangement) and do NOT
   state a technical name -- naming an unconfirmed constituent as fact is hallucination.
4. INTERPRETATION -- reasoning about a feature's significance or behavior. This is the most
   error-prone level, so it is constrained:
   - Interpretation MAY cover: fracture mode (brittle vs ductile), feature morphology, phase
     continuity/connectivity, particle or defect distribution (uniform vs clustered), defect
     severity, and spatial relationships between features.
   - Interpretation MUST NOT infer processing history (heat treatment, cooling rate),
     manufacturing / deposition method (e.g. "typical of thermal-sprayed coatings"), composition,
     or mechanical properties (strength, hardness, toughness) -- these are NOT visible in an SEM
     image, and stating them (including via "typical of..." phrasing) is hallucination even if it
     sounds plausible.
   - If the image alone cannot fully confirm an interpretation, say so explicitly in the answer.

## GROUNDING RULE (direct vs technical)
- Directly visible features -- crack, grain, grain boundary, pore, void, particle, agglomerate,
  surface roughness, fracture surface, dimple, cleavage facet, corrosion pit -- need only visual
  evidence. Name them freely when you see them.
- Technical identifications -- carbide network, twin, dendrite, precipitate, secondary phase,
  phase boundary, lamellae, inclusion, specific named phases -- require the caption/summary to
  confirm the term before you state it as fact. If unconfirmed, describe the feature by its
  appearance only and do NOT state the technical name -- do not launder a caption term into a
  confident visual claim.

## STRICT ANTI-HALLUCINATION RULES
- Base every answer ONLY on what is actually visible. Never invent features, phases, grains,
  compositions, measurements, counts, or defects that are not clearly visible.
- Never state something as fact if you are inferring or guessing it. Prefer qualitative,
  directly-observable claims ("the surface shows a porous, granular texture") over
  precise-sounding but unverifiable ones ("47% porosity").
- Prefer qualitative size language ("a few micrometers", "tens of micrometers", "sub-micron").
  Give a numeric size ONLY as a coarse range you can justify by directly laying the scale bar
  against the feature, and say you used the scale bar -- never state a narrow or single-value
  measurement (e.g. "2.7 um"), since that implies a precision the image cannot support.
- Do NOT assign technical meaning to paper-added annotations (arrows, dashed/colored lines,
  labels, boxes). You may note that a mark is present, but do not state what it "represents" or
  "indicates" (e.g. "the dashed line marks a prior grain boundary") -- that meaning comes from the
  caption, not the image, and the trained model will never see the caption.
- Do NOT generate a comparison question unless the image actually contains multiple distinct
  objects or regions to compare. Do NOT generate a counting question unless multiple countable
  objects are actually visible.
- Agreement between your answer and the caption/summary is a GOOD grounding signal, not a
  shortcut -- but never phrase a question so it is answerable from the caption/summary text alone.
- If the image is too ambiguous, low-quality, blurry, or out of focus to support any real
  question, return an empty list. Never fabricate detail to make the image seem clearer than it
  is.

## CONTEXT (reference only -- never quote it back as the answer)
Caption: {subcaption}
Summary: {summary}

## OUTPUT (strict JSON, no markdown fences)
{{"questions": [
  {{"question": "...", "answer": "...",
    "level": "observation | detection | identification | interpretation",
    "feature": "the specific feature this question is about, or \\"none\\" for broad observation",
    "visual_evidence": "one sentence naming the visible evidence that supports the answer"}}
]}}
If nothing in the image supports a real question, return {{"questions": []}}.
"""


def _panel_filename(image_name: str, panel_label: str, n_panels: int) -> str:
    if n_panels == 1 and panel_label == "main":
        return f"{image_name}_single.jpg"
    return f"{image_name}_{panel_label.upper()}.jpg"


def _prefix_of(name: str) -> str:
    m = re.match(r"([a-zA-Z_]+?)img\d+$", name)
    return m.group(1) if m else "img"


def _build_sample(index_path: Path, images_dir: Path, n_total: int, seed: int):
    """Stratified-by-source-prefix sampling, SEM-only, image-must-exist. `index_path` is a JSON
    file: a list of records, each with an `image_name` and a `panels` list; each panel carries
    `visualization_subtype`, `panel`, `subcaption`, `summary`."""
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    available = []
    for rec in data:
        name = rec.get("image_name", "")
        n_panels = len(rec.get("panels", []))
        for p in rec.get("panels", []):
            if p.get("visualization_subtype") != "SEM":
                continue
            fname = _panel_filename(name, p.get("panel", "main"), n_panels)
            if (images_dir / fname).exists():
                available.append({
                    "image_name": name, "panel": p.get("panel", ""), "filename": fname,
                    "prefix": _prefix_of(name),
                    "subcaption": p.get("subcaption", ""), "summary": p.get("summary", ""),
                })

    random.seed(seed)
    by_prefix = {}
    for r in available:
        by_prefix.setdefault(r["prefix"], []).append(r)

    total_available = len(available)
    sample = []
    remainder = n_total
    prefixes = sorted(by_prefix.keys())
    for i, pfx in enumerate(prefixes):
        group = by_prefix[pfx]
        random.shuffle(group)
        if i == len(prefixes) - 1:
            k = remainder
        else:
            k = round(n_total * len(group) / total_available)
            remainder -= k
        sample.extend(group[:k])

    random.shuffle(sample)
    sample = sample[:n_total]
    logger.info(f"Sampled {len(sample)} SEM panels (from {total_available} available); "
                f"source-prefix pool sizes: "
                f"{{{', '.join(f'{p}: {len(g)}' for p, g in by_prefix.items())}}}")
    return sample


def _parse_json_response(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _call_cost_usd(usage_metadata, price_in_per_1m, price_out_per_1m):
    input_tokens = usage_metadata.prompt_token_count or 0
    output_tokens = ((usage_metadata.candidates_token_count or 0)
                      + (usage_metadata.thoughts_token_count or 0))
    cost = ((input_tokens / 1e6) * price_in_per_1m + (output_tokens / 1e6) * price_out_per_1m)
    return cost, input_tokens, output_tokens


def _generate_for_item(client, item, model_name, images_dir, price_in, price_out):
    img_path = images_dir / item["filename"]
    img_bytes = img_path.read_bytes()
    prompt = PROMPT_TEMPLATE.format(
        subcaption=item["subcaption"] or "(none)",
        summary=item["summary"] or "(none)",
    )
    from google.genai import types
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), prompt],
    )
    cost, input_tokens, output_tokens = _call_cost_usd(response.usage_metadata, price_in, price_out)
    parsed = _parse_json_response(response.text)
    questions = parsed.get("questions", [])[:GENERATE_MAX_QUESTIONS]
    cost_per_question = cost / len(questions) if questions else 0.0

    results = []
    for q_index, q in enumerate(questions):
        level = q.get("level", "")
        if level and level not in VALID_LEVELS:
            logger.warning(f"{item['filename']} q{q_index}: off-taxonomy level={level!r} (kept as-is)")
        results.append({
            "image_name": item["image_name"], "panel": item["panel"], "filename": item["filename"],
            "prefix": item["prefix"], "q_index": q_index,
            "cost_usd": round(cost_per_question, 8),
            "question": q.get("question"), "answer": q.get("answer"), "level": level,
            "feature": q.get("feature", ""), "visual_evidence": q.get("visual_evidence", ""),
            "model": model_name, "prompt_version": GENERATE_PROMPT_VERSION,
        })
    return results, {"cost_usd": cost, "input_tokens": input_tokens, "output_tokens": output_tokens}


def cmd_generate(args):
    from google import genai

    index_path = Path(args.index)
    images_dir = Path(args.images_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = _build_sample(index_path, images_dir, args.n, args.seed)
    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("prompt_version") == GENERATE_PROMPT_VERSION:
                        done.add(rec["filename"])
        if done:
            logger.info(f"Resuming: {len(done)} already generated, skipping those.")

    n_ok, n_empty, n_fail, n_q = 0, 0, 0, 0
    total_cost = 0.0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for i, item in enumerate(sample):
            if item["filename"] in done:
                continue
            try:
                results, info = _generate_for_item(
                    client, item, args.model, images_dir, args.price_in, args.price_out)
                for r in results:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                total_cost += info["cost_usd"]
                if not results:
                    n_empty += 1
                    logger.warning(f"[{i+1}/{len(sample)}] EMPTY (ambiguous image) {item['filename']}")
                else:
                    n_ok += 1
                    n_q += len(results)
                    logger.info(f"[{i+1}/{len(sample)}] OK {item['filename']} "
                                f"({len(results)} question(s), running cost ${total_cost:.4f})")
            except Exception as e:
                n_fail += 1
                logger.error(f"[{i+1}/{len(sample)}] FAIL {item['filename']}: {e}")
            time.sleep(args.sleep)

    logger.info(f"Done in {time.time()-t0:.0f}s: {n_ok} images -> {n_q} questions "
                f"({n_q/max(1,n_ok):.1f} avg/image), {n_empty} empty, {n_fail} failed. "
                f"Cost ${total_cost:.4f}. -> {out_path}")


# ============================================================================================
# SUBCOMMAND 2: answer
#
# Adapted from rep300_core.py + run_gemini31_flash.py (the Gemini arm of the rep300 answering
# benchmark). The record schema, resume/retry logic, and the temperature-rejection fallback are
# unchanged. Hardcoded absolute paths and the Azure/Vertex resource identifiers specific to the
# machine this was run on were replaced with CLI arguments.
# ============================================================================================

ANSWER_FORMATS = ("plain", "evidence")
MAX_RETRIES = 5
BACKOFF_BASE = 4
FATAL_SUBSTRINGS = (
    "deploymentnotfound", "unauthorized", "invalid api key", "access denied",
    "does not support image", "not allowed in this deployment", "user_error",
    "permission denied", "was not found",
)
TEMP_REJECT_SUBSTRINGS = ("temperature", "unsupported_parameter", "unsupported value")


def _answer_call(client, model_id, prompt, img_bytes, use_temp):
    """One generate_content call against Gemini/Vertex. Returns (text, in_tokens, out_tokens)."""
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=0) if use_temp else None
    r = client.models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=[
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes)),
        ])],
        config=cfg)
    txt = getattr(r, "text", None) or ""
    u = getattr(r, "usage_metadata", None)
    in_tok = getattr(u, "prompt_token_count", 0) or 0
    # Thinking tokens bill at the output rate; fold them in so cost/volume are not understated
    # for a thinking-capable model.
    out_tok = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return txt, in_tok, out_tok


def _call_with_retry(client, model_id, prompt, img_bytes, state, rec_id):
    """`state["use_temp"]` is learned once per run: if the deployment rejects temperature=0 on
    the first record, every subsequent record uses the provider default instead of re-probing."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            out = _answer_call(client, model_id, prompt, img_bytes, state["use_temp"])
            return out, ("temperature=0" if state["use_temp"] else "provider-default")
        except Exception as e:
            last = e
            msg = str(e).lower()
            if state["use_temp"] and any(s in msg for s in TEMP_REJECT_SUBSTRINGS):
                logger.info("Deployment rejected temperature; using provider default "
                            "for the rest of this run.")
                state["use_temp"] = False
                continue
            if any(s in msg for s in FATAL_SUBSTRINGS):
                raise
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"{type(e).__name__} on {rec_id} (attempt {attempt+1}/{MAX_RETRIES}), "
                           f"retrying in {wait}s: {e}")
            time.sleep(wait)
    raise last


def cmd_answer(args):
    from google import genai

    benchmark_path = Path(args.benchmark)
    images_dir = Path(args.images_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format not in ANSWER_FORMATS:
        raise SystemExit(f"--format must be one of {ANSWER_FORMATS}")

    records = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
        logger.info(f"Resuming: {len(done)} of {len(records)} already done")

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    state = {"use_temp": True}
    n_ok, n_fail = 0, 0
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as out:
        for i, r in enumerate(records, start=1):
            if r["id"] in done:
                continue
            prompt_key = f"prompt_{args.format}"
            prompt = r[prompt_key]
            img_bytes = (images_dir / r["image"]).read_bytes()
            try:
                t_call = time.time()
                (pred, in_tok, out_tok), decoding = _call_with_retry(
                    client, args.model_id, prompt, img_bytes, state, r["id"])
                elapsed = time.time() - t_call
                out.write(json.dumps({
                    # Shared record schema -- kept identical across every provider arm so
                    # scores are directly comparable. See README for the field meanings.
                    "id": r["id"], "image": r["image"], "filename": r.get("filename"),
                    "split": r.get("split"), "level": r.get("level"), "feature": r.get("feature"),
                    "cluster_id": r.get("cluster_id"), "centrality": r.get("centrality"),
                    "question": r.get("question"), "prompt_format": args.format, "prompt": prompt,
                    "reference": r.get("reference_answer"),
                    "reference_evidence": r.get("reference_evidence"),
                    "prediction": pred, "generation_seconds": round(elapsed, 3),
                    "model": args.model_key, "model_version": args.model_id,
                    "surface": "vertex_gemini", "decoding": decoding,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                }, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
                if n_ok % 25 == 0 or i == len(records):
                    rate = n_ok / max(1e-9, time.time() - t0)
                    logger.info(f"[{i}/{len(records)}] {n_ok} ok, {n_fail} failed, "
                                f"{rate*60:.1f}/min")
            except Exception as e:
                n_fail += 1
                logger.error(f"[{i}/{len(records)}] {r['id']}: FAILED {type(e).__name__}: {e}")
            time.sleep(args.sleep)

    logger.info(f"Done in {(time.time()-t0)/60:.1f} min: {n_ok} ok, {n_fail} failed. -> {out_path}")
    if n_fail:
        logger.info(f"{n_fail} records failed; re-run the same command to retry only those.")


# ============================================================================================
# CLI
# ============================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate QA pairs over a set of SEM micrographs")
    g.add_argument("--index", required=True,
                   help="JSON file: list of {image_name, panels:[{visualization_subtype, "
                        "panel, subcaption, summary}]}")
    g.add_argument("--images-dir", required=True, help="directory containing the panel images")
    g.add_argument("--out", required=True, help="output JSONL path (appended to; resumable)")
    g.add_argument("--project", required=True, help="GCP project for Vertex AI")
    g.add_argument("--location", default="us-central1", help="Vertex AI location")
    g.add_argument("--model", default="gemini-2.5-flash")
    g.add_argument("--n", type=int, default=100, help="number of images to sample and generate for")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--sleep", type=float, default=1.5)
    g.add_argument("--price-in", type=float, default=0.30, help="USD per 1M input tokens")
    g.add_argument("--price-out", type=float, default=2.50, help="USD per 1M output tokens")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("answer", help="run a model over a frozen benchmark and write predictions")
    a.add_argument("--benchmark", required=True,
                   help="JSON file: list of records with id, image, prompt_plain, "
                        "prompt_evidence, reference_answer, reference_evidence, level, "
                        "feature, cluster_id, centrality, split")
    a.add_argument("--images-dir", required=True)
    a.add_argument("--out", required=True, help="output JSONL path (appended to; resumable)")
    a.add_argument("--format", required=True, choices=ANSWER_FORMATS)
    a.add_argument("--project", required=True, help="GCP project for Vertex AI")
    a.add_argument("--location", default="global",
                   help="Vertex AI location; 'global' is what worked for the flash-lite family "
                        "in this project when every regional endpoint returned 404")
    a.add_argument("--model-id", default="gemini-3.1-flash-lite",
                   help="the model actually served; may differ from the requested arm name")
    a.add_argument("--model-key", default="gemini-3.1-flash",
                   help="the arm name to record in output (kept separate from --model-id so a "
                        "provider-side substitution stays visible in the data)")
    a.add_argument("--limit", type=int, default=None, help="first N records only (pilot)")
    a.add_argument("--sleep", type=float, default=0.4)
    a.set_defaults(func=cmd_answer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
