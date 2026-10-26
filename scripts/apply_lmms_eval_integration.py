#!/usr/bin/env python3
"""Install the evaluated OTFS adapters into a clean LMMs-Eval checkout."""

import argparse
import shutil
import subprocess
from pathlib import Path

BASE_COMMIT = "ee31b5a65ea79c8a2a4c9d6fcab736751d39513d"
FILE_MAP = {
    "qwen2_load_video.py": "lmms_eval/models/model_utils/qwen2_load_video.py",
    "qwen2_5_vl_simple.py": "lmms_eval/models/simple/qwen2_5_vl.py",
    "qwen3_vl_simple.py": "lmms_eval/models/simple/qwen3_vl.py",
    "qwen2_5_vl_chat.py": "lmms_eval/models/chat/qwen2_5_vl.py",
    "qwen3_vl_chat.py": "lmms_eval/models/chat/qwen3_vl.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lmms_root", type=Path, help="clean LMMs-Eval checkout")
    args = parser.parse_args()

    lmms_root = args.lmms_root.resolve()
    integration = Path(__file__).resolve().parents[1] / "integrations" / "lmms_eval"
    head = subprocess.check_output(
        ["git", "-C", str(lmms_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != BASE_COMMIT:
        raise RuntimeError(f"expected LMMs-Eval {BASE_COMMIT}, found {head}")

    for source_name, relative_target in FILE_MAP.items():
        source = integration / source_name
        target = lmms_root / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    pyproject = lmms_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines(keepends=True) if '"pywsd"' not in line]
    pyproject.write_text("".join(lines), encoding="utf-8")
    shutil.copy2(integration / "uv.lock", lmms_root / "uv.lock")

    print(f"Applied OTFS integration to {lmms_root}")
    print(f"Verified base commit: {BASE_COMMIT}")
    print("Next: uv sync --frozen --python 3.11.14 --extra qwen --extra video")


if __name__ == "__main__":
    main()
