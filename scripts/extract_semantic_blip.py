#!/usr/bin/env python3
"""Add 1-fps BLIP-ITM question/frame probabilities to an OTFS manifest."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from decord import VideoReader, cpu
from lavis.models import load_model_and_preprocess
from PIL import Image

from otfs.evidence import one_fps_indices
from otfs.io import dump_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list with video_path and question")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-root", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    items = load_json(args.manifest)
    model, visual_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=args.device, is_eval=True
    )
    output = []
    for number, item in enumerate(items, 1):
        path = Path(args.video_root) / item["video_path"]
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
        indices = one_fps_indices(len(reader), float(reader.get_avg_fps()))
        question = text_processors["eval"](item["question"])
        scores = []
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                frames = reader.get_batch(indices[start : start + args.batch_size]).asnumpy()
                images = torch.stack(
                    [visual_processors["eval"](Image.fromarray(frame)) for frame in frames]
                ).to(args.device)
                logits, _ = model(
                    {"image": images, "text_input": [question] * len(images)}, match_head="itm-e"
                )
                scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        updated = dict(item)
        updated["frame_indices"] = indices
        updated.setdefault("evidence", {})["semantic"] = scores
        output.append(updated)
        print(f"[{number}/{len(items)}] {path}")
    dump_json(output, args.output)


if __name__ == "__main__":
    main()
