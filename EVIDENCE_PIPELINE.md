# Constructing OTFS evidence

Run commands from the OTFS repository root. Create the Python 3.10 `otfs`
environment from the [root README](README.md#environment) first. Semantic
extraction uses a separate AKS/SeViLA environment; semantic and transition
extraction require a GPU by default and download model checkpoints on first use.
Benchmark data, evidence, and selection results are not bundled.

## Dataset preparation

Download the benchmarks from
[Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME),
[LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench),
and [MLVU](https://github.com/JUNJIE99/MLVU). Recommended local paths:

| Dataset argument | Annotation JSON | Video root | Paper items |
|---|---|---|---:|
| `videomme` | `data/videomme/videomme.json` | `data/videomme/videos` | 2700 |
| `longvideobench` | `data/longvideobench/lvb_val.json` | `data/longvideobench/videos` | 1337 |
| `mlvu` | `data/mlvu/mlvu.json` | `data/mlvu/videos` | 2174 |

Annotations must be JSON lists. Each item needs `question` and these fields:

| Dataset | Evidence fields | Video path relative to video root |
|---|---|---|
| Video-MME | `videoID`, `options` | `<videoID>.mp4` |
| LongVideoBench | `video_path`, `candidates`, `id` | `video_path` |
| MLVU | `video`, `candidates` | `video` |

Use LongVideoBench's `lvb_val.json` directly. For Video-MME, convert
`test-00000-of-00001.parquet` to JSON without reordering rows, or use
[AKS's converted annotation](https://github.com/ncTimTang/AKS/blob/main/datasets/videomme/videomme.json).
For MLVU, concatenate the seven multiple-choice development tasks in evaluation
order into `mlvu.json`, retaining `question_type`; exclude open-ended tasks.

Preserve original IDs, category fields, and answers (`correct_choice` for
LongVideoBench, `answer` for Video-MME/MLVU). Use the same annotation JSON and
row order for evidence preparation and downstream evaluation, not a generated
manifest or evidence file.

Paths are relative to the repository root; external datasets can use absolute
paths. Use the same video root for extraction and evaluation. Scripts do not
download benchmark data or annotations.

The commands below use Video-MME. For another dataset, change these values
using the path table above and keep a separate `WORK_DIR`:

```bash
export DATASET=videomme
export ANNOTATIONS=data/videomme/videomme.json
export VIDEO_ROOT=data/videomme/videos
export WORK_DIR="outputs/$DATASET"
```

## 1. Semantic evidence

Install and activate the feature-extraction environment following
[AKS's Feature Extraction instructions](https://github.com/ncTimTang/AKS#feature-extraction).
Use its SeViLA/LAVIS installation and replace
`lavis/models/blip_models/blip_image_text_matching.py` with AKS's file, as
instructed there: the `itm-e` interface below requires that modification.
Run this step in that environment; do not install `requirements-evidence.txt`
there. The script computes BLIP-ITM-large question/frame probabilities on
AKS's 1-fps candidate stream.

```bash
python scripts/prepare_manifest.py \
  --dataset "$DATASET" \
  --annotations "$ANNOTATIONS" \
  --output "$WORK_DIR/manifest.json"

python scripts/extract_semantic_blip.py \
  --manifest "$WORK_DIR/manifest.json" \
  --video-root "$VIDEO_ROOT" \
  --output "$WORK_DIR/semantic.json"
```

## 2. Utility evidence

Switch to `otfs` and obtain the official [DIG](https://github.com/Jialuo-Li/DIG)
sparse rewards. Clone once and reuse the checkout for all datasets:

```bash
conda activate otfs
git clone https://github.com/Jialuo-Li/DIG.git third_party/DIG
git -C third_party/DIG checkout ffe6d2554e2c14bb1905ce5ad66ff13ba8cd5da4

python scripts/interpolate_dig_rewards.py \
  --manifest "$WORK_DIR/semantic.json" \
  --dig-rewards "third_party/DIG/rewards/Qwen2.5-VL-7B-Instruct_${DATASET}.json" \
  --output "$WORK_DIR/semantic_utility.json"
```

OTFS linearly interpolates sparse rewards onto the candidate stream; global
queries receive zero utility. To regenerate sparse rewards or use new data,
follow DIG's Query Identification, CAFS, and Reward Assignment workflow and
pass its output to `--dig-rewards`.

## 3. Transition evidence

Use `otfs` to compute adjacent-frame cosine distances with
`facebook/dinov2-base`. The model is reused across videos and scores are cached
per video within the run:

```bash
conda activate otfs
pip install -r requirements-evidence.txt

python scripts/extract_transition_dinov2.py \
  --manifest "$WORK_DIR/semantic_utility.json" \
  --video-root "$VIDEO_ROOT" \
  --output "$WORK_DIR/evidence.json"
```

Global queries keep zero transition scores for paper reproduction. Use
`--compute-global` only for an ablation.

The final `$WORK_DIR/evidence.json` is ready for
[frame selection](README.md#2-select-frames). Repeat for the other datasets;
keep generated files under `outputs/` and record code/checkpoint revisions and
package versions with your results.
