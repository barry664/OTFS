# Reproducing downstream evaluation with LMMs-Eval

Requires Linux, Git, NVIDIA GPUs, benchmark videos/annotations, and a local
Qwen2.5-VL-7B-Instruct or Qwen3-VL-4B-Instruct checkpoint. First generate
evidence and selections following the [root README](../../README.md).
The example below uses Qwen2.5-VL-7B, Video-MME, and 32 frames.

## 1. Create the environment

Run from the OTFS repository root. `LMMS_ROOT` must be a new checkout path:

```bash
python -m pip install uv
export OTFS_ROOT="$PWD"
export LMMS_ROOT=/absolute/path/to/lmms-eval-otfs

bash integrations/lmms_eval/bootstrap.sh "$LMMS_ROOT"
source "$LMMS_ROOT/.venv/bin/activate"
```

This installs the pinned LMMs-Eval v0.6.1 checkout, OTFS adapters, and locked
Python 3.11 environment. See [verified versions](VERIFIED_ENVIRONMENT.md).

## 2. Set benchmark locations

Use the same videos as evidence extraction. Export absolute video-directory
paths in the inference shell; replace these values if your data is elsewhere:

```bash
export OTFS_VIDEOMME_VIDEO_DIR="$OTFS_ROOT/data/videomme/videos"
export OTFS_LONGVIDEOBENCH_VIDEO_DIR="$OTFS_ROOT/data/longvideobench/videos"
export OTFS_MLVU_VIDEO_DIR="$OTFS_ROOT/data/mlvu/videos"
```

Each variable points directly to the video directory; no extra directory is
appended. Original annotation JSONs are passed separately to `--annotations`:

| Dataset argument | Original annotation JSON |
|---|---|
| `videomme` | `data/videomme/videomme.json` |
| `longvideobench` | `data/longvideobench/lvb_val.json` |
| `mlvu` | `data/mlvu/mlvu.json` |

Paths in the table are relative to the OTFS root. Use the same annotations and
row order as evidence preparation, not generated manifest/evidence files.
See [dataset preparation](../../EVIDENCE_PIPELINE.md#dataset-preparation)
for sources and required fields.

## 3. Build an isolated task

From the OTFS root, combine the original annotation with the detailed selection
JSON containing `selected_frames` (not `*_simple.json`):

```bash
python scripts/prepare_lmms_eval_task.py \
  --dataset videomme \
  --budget 32 \
  --annotations data/videomme/videomme.json \
  --selections outputs/selections/videomme_32.json \
  --output-dir eval_tasks/videomme_32
```

This writes merged `annotations.json`, a `task/` directory, and `manifest.json`
with input/output fingerprints. For other settings, change `--dataset`,
`--budget` (32/64), and all three paths together; use a distinct output directory.

## 4. Run inference

The task name is `otfs_<dataset>_<budget>`. Example on one 8-GPU node:

```bash
export MODEL_PATH=/absolute/path/to/Qwen2.5-VL-7B-Instruct
cd "$LMMS_ROOT"

accelerate launch --multi_gpu --num_processes 8 -m lmms_eval \
  --include_path "$OTFS_ROOT/eval_tasks/videomme_32/task" \
  --model qwen2_5_vl \
  --model_args pretrained="$MODEL_PATH",use_topk=True,nframes=32 \
  --tasks otfs_videomme_32 \
  --batch_size 1 --log_samples \
  --output_path "$OTFS_ROOT/results/qwen25/videomme_32"
```

For Qwen3-VL-4B, change `--model` to `qwen3_vl` and `pretrained` to the
Qwen3-VL-4B-Instruct checkpoint. For another task, change `--include_path`,
`--tasks`, `nframes`, and `--output_path` together.

## 5. LongVideoBench scoring

For LongVideoBench, recompute strict multiple-choice accuracy from the sample log:

```bash
python "$OTFS_ROOT/scripts/eval_longvideobench_accuracy.py" \
  /absolute/path/to/the_samples_log.jsonl
```

Video-MME and MLVU use LMMs-Eval's metrics. Retain sample logs, task manifests,
result JSONs, and model checkpoint revisions with reported scores.
