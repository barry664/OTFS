#!/usr/bin/env python3
"""Convert benchmark annotations to the canonical OTFS manifest."""

import argparse
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otfs.io import dump_json, load_json


def parenthesized_prompt(question: str, choices: list[str]) -> str:
    lines = [question]
    lines.extend(f"({string.ascii_uppercase[i]}) {choice}" for i, choice in enumerate(choices))
    return "\n".join(lines)


def convert(item: dict, dataset: str) -> dict:
    if dataset == "videomme":
        video_id = item["videoID"]
        video_path = video_id + ".mp4"
        reward_question = "\n".join([item["question"]] + [str(x).strip() for x in item["options"]])
        question_id = item.get("question_id", "")
    elif dataset == "longvideobench":
        video_id = item.get("video_id", item["video_path"])
        video_path = item["video_path"]
        reward_question = parenthesized_prompt(item["question"], item["candidates"])
        question_id = item.get("id", "")
    elif dataset == "mlvu":
        video_id = item["video"]
        video_path = item["video"]
        reward_question = parenthesized_prompt(item["question"], item["candidates"])
        question_id = item.get("question_id", "")
    else:
        raise ValueError(dataset)
    return {
        "video_id": video_id,
        "question_id": question_id,
        "video_path": video_path,
        "reward_video_path": video_path,
        "question": item["question"],
        "reward_question": reward_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("videomme", "longvideobench", "mlvu"))
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    annotations = load_json(args.annotations)
    output = [convert(item, args.dataset) for item in annotations]
    dump_json(output, args.output)
    print(f"Wrote {len(output)} {args.dataset} manifest items to {args.output}")


if __name__ == "__main__":
    main()
