"""Input/output helpers for the canonical OTFS evidence schema."""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .core import OTFSConfig, otfs_select, prepare_paper_evidence


def load_json(path: str | Path) -> Any:
    source = Path(path)
    with _open_text(source, "rt") as handle:
        return json.load(handle)


def _open_text(path: Path, mode: str):
    """Open plain JSON or a gzip-compressed JSON file as UTF-8 text."""

    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode.replace("t", ""), encoding="utf-8")


def iter_json_array(path: str | Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream items from a top-level JSON array without loading it all in memory."""

    decoder = json.JSONDecoder()
    with _open_text(Path(path), "rt") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected a top-level JSON array: {path}")
                position += 1
                started = True
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            remaining = buffer[position:]
            try:
                value, used = decoder.raw_decode(remaining)
                yield value
                position += used
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = remaining + handle.read(chunk_size)
                position = 0


def dump_json(value: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(output, "wt") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def evidence_from_item(item: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    """Read either the canonical evidence schema or the original experiment schema."""

    if "evidence" in item:
        atoms = item["evidence"]
        semantic = atoms["semantic"]
        utility = atoms["utility"]
        transition = atoms["transition"]
    else:
        semantic = item["semantic_anchors"]["question_scores"]
        utility = item["dgi_rewards"]
        transition = item["frame_diffs"]
    prepared = prepare_paper_evidence(semantic, utility, transition)
    indices = item.get("frame_indices", list(range(prepared.shape[1])))
    return prepared, [int(x) for x in indices]


def select_dataset(items: list[dict[str, Any]], config: OTFSConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        evidence, frame_indices = evidence_from_item(item)
        result = otfs_select(evidence, frame_indices, config, evidence_is_prepared=True)
        results.append(
            {
                "video_id": item.get("video_id", item.get("videoID", item.get("video", ""))),
                "question_id": item.get("question_id", item.get("id", "")),
                "question": item.get("question", ""),
                "selected_frames": result.selected_frame_indices,
                "selected_positions": result.selected_positions,
                "frame_weights": result.frame_weights.tolist(),
                "source_mass": result.transport_plan.sum(axis=1).tolist(),
                "iterations_run": result.iterations_run,
            }
        )
    return results
