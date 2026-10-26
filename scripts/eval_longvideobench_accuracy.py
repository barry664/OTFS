#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

INDEX_TO_OPTION = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
STRICT_PATTERNS = [
    re.compile(r"(?:FINAL\s+ANSWER|ANSWER|ANS|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-E])\)?\b"),
    re.compile(r"^\s*\(?([A-E])\)?\s*[\.:\),\-]", re.MULTILINE),
    re.compile(r"^\s*\(?([A-E])\)?\s*$", re.MULTILINE),
]


def extract_text(filtered_resps: Any) -> str:
    if filtered_resps is None:
        return ""
    if isinstance(filtered_resps, str):
        return filtered_resps
    if isinstance(filtered_resps, list):
        parts = []
        for item in filtered_resps:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(filtered_resps, ensure_ascii=False)


def parse_prediction(filtered_resps: Any) -> Optional[str]:
    text = extract_text(filtered_resps).upper()
    for pattern in STRICT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def parse_target(target: Any) -> Optional[str]:
    if isinstance(target, int):
        return INDEX_TO_OPTION.get(target)
    if isinstance(target, str):
        target = target.strip()
        if target.isdigit():
            return INDEX_TO_OPTION.get(int(target))
        target = target.upper()
        if target in INDEX_TO_OPTION.values():
            return target
    return None


def evaluate(path: Path, show_errors: int) -> None:
    total = 0
    correct = 0
    invalid_pred = 0
    invalid_target = 0
    error_examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            pred = parse_prediction(row.get("filtered_resps"))
            gold = parse_target(row.get("target"))

            if gold is None:
                invalid_target += 1
                continue

            total += 1
            if pred is None:
                invalid_pred += 1
            if pred == gold:
                correct += 1
            elif len(error_examples) < show_errors:
                error_examples.append(
                    {
                        "line": line_no,
                        "pred": pred,
                        "gold": gold,
                        "filtered_resps": extract_text(row.get("filtered_resps")),
                    }
                )

    accuracy = (correct / total) if total else 0.0
    valid_pred = total - invalid_pred

    print(f"file: {path}")
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.4%}")
    print(f"valid_pred: {valid_pred}")
    print(f"invalid_pred: {invalid_pred}")
    print(f"invalid_target: {invalid_target}")

    if error_examples:
        print("\nerror_examples:")
        for item in error_examples:
            print(json.dumps(item, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LongVideoBench QA accuracy from JSONL inference results."
    )
    parser.add_argument("input", type=Path, help="Path to the JSONL result file")
    parser.add_argument(
        "--show-errors",
        type=int,
        default=5,
        help="Number of wrong examples to print",
    )
    args = parser.parse_args()
    evaluate(args.input, args.show_errors)


if __name__ == "__main__":
    main()
