# SEM-VQA

Visual question answering over scanning electron micrographs (SEM) of materials-science
specimens: a corpus-generation pipeline, a held-out evaluation benchmark built from it, and
a two-stage LoRA adaptation of Qwen3.5-2B trained and evaluated on it. Stage 1 teaches
image-to-summary description, stage 2 teaches VQA. Six stage-2 cells are compared: three
adapter initialisations x two supervision targets.

## Dataset

The released data is hosted on Hugging Face:

- **[SEM-VQA](https://huggingface.co/datasets/uhbuvbuvu/sem-vqa)** — the full corpus: 224,026
  QA pairs over 47,981 SEM image panels (16,192 base images), spanning four reasoning levels
  (observation, detection, identification, interpretation) across ceramics/production,
  Ni-based alloy, and composite specimens. Train/test splits of 198,764 / 23,763 QA pairs are
  provided, plus a 1,499-QA benchmark pool held out and curated separately into SEM-VQA-Bench.
  Every image is sourced from peer-reviewed, open-access materials-science articles. CC BY 4.0.
- **[SEM-VQA-Bench](https://huggingface.co/datasets/uhbuvbuvu/sem-vqa-bench)** — the frozen
  300-image / ~1,500-QA evaluation benchmark produced by `benchmarking/` (see below), with
  both prompt formats, reference answers/evidence, and DINOv3 cluster/centrality metadata
  baked in. CC BY 4.0.

Download one or both to populate `--data-dir` for training (`SEM-VQA`) and `--split
benchmark` evaluation (`SEM-VQA-Bench`).

## Repository layout

```
corpus_generation/   QA-pair generation over SEM micrographs, and the Gemini arm of the
                     benchmark-answering pipeline
benchmarking/        construction of the held-out 300-image / 1,497-QA evaluation benchmark
train/               LoRA training, inference and evaluation for both stages
```

## Environment

`transformers>=5.x` is mandatory. Qwen3.5 is multimodal and must load via
`AutoModelForImageTextToText`; `AutoModelForCausalLM` matches zero keys and silently returns
a randomly initialised network. See `train/requirements.txt` for exact pinned versions,
`benchmarking/requirements.txt` for the benchmark-construction dependencies, and
`corpus_generation/sem_vqa_toolkit.py`'s docstring for its single dependency (`google-genai`).

## Corpus generation

`corpus_generation/sem_vqa_toolkit.py` holds the two pipelines used to build and evaluate the
SEM-VQA corpus, as two subcommands of one script:

- `generate` — QA-pair generation over SEM micrographs (prompt version
  `v7_four_level_grounded`, gemini-2.5-flash via Vertex AI). This is the exact prompt and
  pipeline logic used for the raw generation run (309,518 QA pairs over 66,385 SEM panels,
  before content-deduplication and quality filtering); the released **SEM-VQA** corpus
  (224,026 QA pairs over 47,981 panels — see "Dataset" above) is that run after
  `benchmarking/step3a_unique_images.py`-style dedup and QC.
- `answer` — runs a model over a frozen benchmark and writes predictions in the schema used
  for scoring. This is the Gemini arm (gemini-3.1-flash-lite) of the rep300 answering
  benchmark; the other proprietary arms used a different SDK each and are not included, so
  this script's only dependency is `google-genai`.

Both subcommands authenticate to Vertex AI via Application Default Credentials
(`gcloud auth application-default login`); no API key is read or stored anywhere in the file.

```
python corpus_generation/sem_vqa_toolkit.py generate --index index.json --images-dir ./images \
    --out sem_vqa.jsonl --project YOUR_GCP_PROJECT --n 100

python corpus_generation/sem_vqa_toolkit.py answer --benchmark rep300/benchmark_rep300.json \
    --images-dir rep300/images --out predictions.jsonl --format evidence --project YOUR_GCP_PROJECT
```

## Data

Training reads `stage1_{train,val,test}.jsonl` (image -> summary) and
`stage2_{train,val,test,benchmark}.jsonl` (image + question -> answer, carrying
`visual_evidence` for Format B) from `--data-dir`.

The partition is image-level, seed 42, drawn over the 47,681 non-benchmark images:
42,989 train / 2,345 val / 2,347 test, plus the 300 benchmark images held out entirely
(see "Benchmark construction" below). Stage 2 covers 200,675 / 10,935 / 10,917 / 1,499 QA
records. The split definitions ship as JSON alongside the dataset; the scripts that
generated them are not part of this archive.

## Stage 1 — image to summary

```
python train/train_stage1_summary.py --model <qwen3.5-2b> --data-dir <data> \
    --output-dir runs/stage1_frozen \
    --epochs 2 --batch-size 16 --grad-accum 2 --lr 1e-4 \
    --logging-steps 20 --eval-steps 200 --save-steps 200 --save-total-limit 3

python train/train_stage1_summary.py ... --output-dir runs/stage1_train_vision --train-vision
```

`--train-vision` adds LoRA to the vision tower (284 adapted modules vs 186).

## Stage 2 — visual question answering

The vision tower is frozen in every stage-2 run. Cells differ only in where the adapter
starts and what it is trained to emit.

| Cell | Init | Target |
|---|---|---|
| B1-A | *omit* `--init-adapter` | `--target-format answer` |
| B2-A | `--init-adapter runs/stage1_frozen/final` | `--target-format answer` |
| B3-A | `--init-adapter runs/stage1_train_vision/final` | `--target-format answer` |
| B1-B | *omit* `--init-adapter` | `--target-format evidence_answer` |
| B2-B | `--init-adapter runs/stage1_frozen/final` | `--target-format evidence_answer` |
| B3-B | `--init-adapter runs/stage1_train_vision/final` | `--target-format evidence_answer` |

```
python train/train_stage2_vqa.py --output-dir runs/b1_fmtB --target-format evidence_answer
python train/train_stage2_vqa.py --output-dir runs/b2_fmtB --target-format evidence_answer \
    --init-adapter runs/stage1_frozen/final
```

The system prompt and the target move together as one `--target-format` switch. Format A
trains on the answer alone; Format B on `Evidence: ...\nAnswer: ...`.

## Evaluation

`--target-format` must match how the adapter was trained.

```
python train/eval_stage2.py --adapter runs/b1_fmtB/final --target-format evidence_answer --split benchmark --gen-n -1
```

Format B predictions have their `Answer:` span extracted into `pred`; the untouched
generation is kept as `pred_full`.

## Hyperparameters

LoRA r=16, alpha=32, dropout 0.05, no bias, no RSLoRA. AdamW (fused), betas 0.9/0.999,
eps 1e-8, weight decay 0, cosine schedule, grad clip 1.0, bfloat16, SDPA, gradient
checkpointing, seed 42. Batch 16 x grad-accum 2 = effective 32.

Stage 1: 2 epochs, lr 1e-4, max_length 1536, max_pixels 1024^2, 2,688 steps, 80 warmup.
Stage 2: 1 epoch, lr 7e-5, max_length 1024, max_pixels 512^2, 6,272 steps, 188 warmup.
Min pixels 448^2 in both.

Loss is token-level cross-entropy on the target span only; the collator masks every prompt
position with -100, so image tokens, system prompt and question contribute no gradient.

## Format-A control

`train/eval_control_run_all.sh` + `train/eval_control_summarize.py` re-evaluate the three
Format-A adapters under the evidence prompt, isolating prompt format from training target.

## Benchmark construction

`benchmarking/` selects 300 representative SEM images from the corpus and freezes them into
a benchmark of 1,497 QA pairs with both prompt formats baked in. This is the pipeline that
produced the released **[SEM-VQA-Bench](https://huggingface.co/datasets/uhbuvbuvu/sem-vqa-bench)**
dataset (see "Dataset" above).

Inputs:
- `sem_vqa_corpus.jsonl` — 224,026 QA records over 47,981 images
- `images/` — the source images, named by the corpus `filename` field

Run order:

```
python benchmarking/step1_extract_embeddings.py   --eval-files sem_vqa_corpus.jsonl --image-dir images --output-dir clusters
python benchmarking/step2_cluster.py              --input-dir clusters --method kmeans --k 5
python benchmarking/step2a_sweep_k.py             --embeddings-dir clusters --k-min 3 --k-max 10
python benchmarking/step2b_export_assignments.py  --embeddings-dir clusters
python benchmarking/step3_select_300.py           --embeddings-dir clusters --qa-source sem_vqa_corpus.jsonl
python benchmarking/step3a_unique_images.py       --image-dir images --corpus sem_vqa_corpus.jsonl --output-dir .
python benchmarking/step4_build_benchmark.py      --subset-json clusters/representative_subset_300.json \
                                                   --source-jsonl sem_vqa_corpus.jsonl --image-dir images --output-dir rep300
```

Step 1 needs a GPU. Steps 2–4 are CPU-only. Steps 2a and 3a are optional: 2a produces the k
sweep, 3a produces the filename-alias table.

Output: `rep300/benchmark_rep300.json` — 1,497 records, each carrying `prompt_plain`,
`prompt_evidence`, `reference_answer`, `reference_evidence`, `level`, `feature`,
`cluster_id`, `centrality`, `distance_to_centroid`, `md5`; `rep300/images/` — the 300
selected images.

Method: DINOv3 ViT-B/16 CLS token → L2-normalise → PCA(50) → k-means(k=5), all seed 42.
Within each cluster, images are ranked by distance to centroid and banded core / mid /
peripheral at the 10th and 70th percentiles. 20 images are drawn from each of the 15
(cluster × band) cells by greedy coverage of question level and feature, breaking ties on
the largest distance gap and then on seeded random. Step 2 defaults to HDBSCAN; pass
`--method kmeans --k 5` for the reported run.

## Not included

`download.py` and the environment bootstrap script are omitted; both hardcode paths
specific to the machines they ran on. The split-construction scripts (`build_splits.py`,
`build_stage2_vqa.py`) are omitted for the same reason.

## License

The **SEM-VQA** and **SEM-VQA-Bench** datasets are released under CC BY 4.0 (see their
dataset cards, linked under "Dataset" above).
