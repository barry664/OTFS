#!/usr/bin/env python3
"""Add DINOv2 transition evidence to a canonical OTFS manifest."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otfs.evidence import Dinov2TransitionExtractor
from otfs.io import dump_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list with video_path fields")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-global",
        action="store_true",
        help="Also compute transition evidence for DIG query_type=global. The paper path leaves it zero.",
    )
    args = parser.parse_args()

    items = load_json(args.manifest)
    extractor = None
    cache = {}
    output = []
    for number, item in enumerate(items, 1):
        video_path = Path(args.video_root) / item["video_path"]
        cache_key = str(video_path.resolve())
        is_global = str(item.get("query_type", "")).lower() == "global"
        if is_global and not args.compute_global:
            indices = item.get("frame_indices")
            if indices is None:
                raise ValueError(
                    "global items require frame_indices from semantic extraction; "
                    "run extract_semantic_blip.py first"
                )
            scores = [0.0] * len(indices)
        else:
            if cache_key not in cache:
                if extractor is None:
                    extractor = Dinov2TransitionExtractor(device=args.device)
                cache[cache_key] = extractor.extract(video_path, batch_size=args.batch_size)
            indices, scores = cache[cache_key]
        updated = dict(item)
        updated["frame_indices"] = indices
        updated.setdefault("evidence", {})["transition"] = scores
        output.append(updated)
        print(f"[{number}/{len(items)}] {video_path}")
    dump_json(output, args.output)


if __name__ == "__main__":
    main()
