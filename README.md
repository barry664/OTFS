# OTFS

Official reproduction code for our NeurIPS 2026 paper, [**A Principled Optimal
Transport Framework for Frame Selection in Long Video Understanding**](https://arxiv.org/pdf/2610.27692).

OTFS is a training-free frame selector. It allocates three heterogeneous
evidence sources—semantic relevance, answer-oriented utility, and temporal
transition—to a 1-fps candidate-frame stream with semi-unbalanced entropic
optimal transport. The target marginal is converted to an exact size-`k`
subset by coverage-aware dynamic programming.

## Repository layout

```text
otfs/                     optimization, evidence, and JSON I/O
scripts/                  evidence, selection, evaluation utilities
integrations/lmms_eval/   exact frame-index evaluation adapter snapshot
EVIDENCE_PIPELINE.md      data preparation and evidence construction guide
frame_select_otfs.py      directly executable frame-selection entry point
```

## Environment

Clone the repository if needed, then create the environment:

```bash
git clone https://github.com/barry664/OTFS.git
cd OTFS
conda create -n otfs python=3.10 -y
conda activate otfs
pip install -r requirements.txt
```

Benchmark reproduction also requires the separate semantic and transition extraction environments described
in [EVIDENCE_PIPELINE.md](EVIDENCE_PIPELINE.md).

## Reproduce benchmark experiments

### 1. Construct the evidence

Follow [EVIDENCE_PIPELINE.md](EVIDENCE_PIPELINE.md) to prepare each benchmark
dataset and construct semantic, utility, and transition evidence. This requires
model checkpoints and a GPU; utility construction uses external DIG sparse
rewards. The final files will be `outputs/videomme/evidence.json`,
`outputs/longvideobench/evidence.json`, and `outputs/mlvu/evidence.json`.

### 2. Select frames

For one dataset and budget, activate the CPU environment and run:

```bash
conda activate otfs
python frame_select_otfs.py \
  --dataset videomme \
  --budget 32 \
  --input outputs/videomme/evidence.json \
  --output outputs/selections/videomme_32.json
```

The command writes a detailed JSON with `selected_frames`, sample metadata,
and frame weights, plus a `_simple.json` containing only frame-index lists.
Use the detailed JSON for downstream evaluation.

After constructing evidence for all three datasets, generate all six
dataset-budget combinations with:

```bash
bash scripts/run_paper_selection.sh \
  outputs/videomme/evidence.json \
  outputs/longvideobench/evidence.json \
  outputs/mlvu/evidence.json
```

### 3. Evaluate the selected frames

Follow [integrations/lmms_eval/README.md](integrations/lmms_eval/README.md) to
run downstream question answering and scoring with your generated detailed
selection files. OTFS selects frames; downstream
question answering uses Qwen2.5-VL-7B or Qwen3-VL-4B in the paper's setup.

## Citation

Please consider citing our work as follows if it is helpful.

```
TBD
```

## Acknowledgments

OTFS builds on AKS, DIG, LAVIS, DINOv2, LMMs-Eval, and Qwen-VL-Utils. We thanks these authors for their efforts!
