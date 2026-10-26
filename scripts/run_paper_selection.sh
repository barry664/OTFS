#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <videomme-evidence.json[.gz]> <longvideobench-evidence.json[.gz]> <mlvu-evidence.json[.gz]> [output-dir]" >&2
  exit 2
fi

inputs=("$1" "$2" "$3")
datasets=(videomme longvideobench mlvu)
output_dir="${4:-$ROOT_DIR/outputs/selections}"
python_bin="${OTFS_PYTHON:-python}"

for input_path in "${inputs[@]}"; do
  if [[ ! -f "$input_path" ]]; then
    echo "Evidence file not found: $input_path. Construct the evidence following EVIDENCE_PIPELINE.md first." >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
for budget in 32 64; do
  for i in "${!datasets[@]}"; do
    "$python_bin" "$ROOT_DIR/frame_select_otfs.py" \
      --dataset "${datasets[$i]}" --budget "$budget" \
      --input "${inputs[$i]}" --output "$output_dir/${datasets[$i]}_${budget}.json"
  done
done
