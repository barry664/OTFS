#!/usr/bin/env python3
"""Build an immutable OTFS annotation and isolated LMMs-Eval task directory."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


YAML_NAMES = {
    "videomme": "videomme.yaml",
    "longvideobench": "longvideobench_val_v.yaml",
    "mlvu": "mlvu_dev.yaml",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(YAML_NAMES))
    parser.add_argument("--budget", required=True, type=int, choices=(32, 64))
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--selections", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-name")
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    selections = load_json(args.selections)
    if len(annotations) != len(selections):
        raise ValueError(
            f"annotation/selection count differs: {len(annotations)} != {len(selections)}"
        )

    merged = []
    for index, (annotation, selection) in enumerate(zip(annotations, selections)):
        if annotation.get("question") != selection.get("question"):
            raise ValueError(f"question mismatch at item {index}")
        frames = selection.get("selected_frames")
        if not isinstance(frames, list):
            raise ValueError(f"missing selected_frames at item {index}")
        expected = min(args.budget, len(selection.get("frame_weights", frames)))
        if len(frames) != expected or len(frames) != len(set(frames)):
            raise ValueError(f"invalid selected-frame count at item {index}")
        updated = dict(annotation)
        updated["frame_idx"] = frames
        updated["frame_num"] = args.budget
        updated["use_topk"] = True
        merged.append(updated)

    output_dir = args.output_dir.resolve()
    task_dir = output_dir / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "annotations.json"
    with annotation_path.open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, ensure_ascii=False, indent=2)

    integration = Path(__file__).resolve().parents[1] / "integrations" / "lmms_eval"
    assets = integration / "tasks" / args.dataset
    shutil.copy2(assets / "utils.py", task_dir / "utils.py")
    if args.dataset == "mlvu":
        shutil.copy2(assets / "mlvu_test.yaml", task_dir / "mlvu_test.yaml")

    task_name = args.task_name or f"otfs_{args.dataset}_{args.budget}"
    template = (assets / "task.yaml.template").read_text(encoding="utf-8")
    rendered = template.replace("__ANNOTATION_PATH__", str(annotation_path))
    rendered = rendered.replace("__TASK_NAME__", task_name)
    yaml_path = task_dir / YAML_NAMES[args.dataset]
    yaml_path.write_text(rendered, encoding="utf-8")

    manifest = {
        "dataset": args.dataset,
        "budget": args.budget,
        "task_name": task_name,
        "num_examples": len(merged),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "selections": str(args.selections.resolve()),
        "selections_sha256": sha256(args.selections),
        "eval_annotations_sha256": sha256(annotation_path),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print(f"Prepared {len(merged)} examples")
    print(f"Task: {task_name}")
    print(f"Include path: {task_dir}")


if __name__ == "__main__":
    main()
