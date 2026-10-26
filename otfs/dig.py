"""Match official DIG reward records to canonical OTFS manifests."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dig_key(item: dict[str, Any]) -> tuple[str, str]:
    return os.path.basename(str(item["video_path"])), normalize_text(item["question"])


def manifest_key(item: dict[str, Any]) -> tuple[str, str]:
    video = item.get(
        "reward_video_path",
        item.get("video_path", item.get("video", item.get("videoID", ""))),
    )
    question = item.get("reward_question", item["question"])
    return os.path.basename(str(video)), normalize_text(question)


def index_rewards(items: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for item in items:
        key = dig_key(item)
        if key in result:
            raise ValueError(f"duplicate DIG reward key: {key}")
        result[key] = item
    return result
