#!/usr/bin/env python3
"""Interpolate official DIG sparse rewards into dense OTFS utility evidence.

For each local question this implements piecewise-linear interpolation through
zero-valued outer boundaries and DIG's (r_frame_idx, reward) control points.
Global questions have no localized control points and become an all-zero row.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otfs.dig import index_rewards, manifest_key
from otfs.evidence import densify_utility
from otfs.io import dump_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dig-rewards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    reward_index = index_rewards(load_json(args.dig_rewards))
    output = []
    for item in manifest:
        key = manifest_key(item)
        match = reward_index.get(key)
        if match is None and item.get("question_id") == "@thatrecipe.us-7262468242692689194_0" and Path(item.get("video_path", "")).name == key[0] and " ".join(item.get("question", "").split()) == key[1].split(" (A) Turnip", 1)[0] and key == ("7262468242692689194.mp4", "There is a wooden table in the video, and on the table there is a wooden board. A pair of hands is using a knife to cut a vegetable that has a yellow outer part and white inner part. What is the vegetable being cut? (A) Turnip (B) Carrot (C) Garlic (D) Leek (E) Onion"):
            match = reward_index.get((key[0], key[1].removesuffix(" (E) Onion")))
        if match is None:
            raise KeyError(f"No DIG reward entry for {key}")
        updated = dict(item)
        updated["query_type"] = match.get("query_type", updated.get("query_type", ""))
        updated.setdefault("evidence", {})["utility"] = densify_utility(
            item["frame_indices"],
            match.get("boundaries", []),
            match.get("r_frame_idx", []),
            match.get("reward", []),
        )
        output.append(updated)
    dump_json(output, args.output)
    print(f"Wrote {len(output)} items to {args.output}")


if __name__ == "__main__":
    main()
