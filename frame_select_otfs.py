#!/usr/bin/env python3
"""Run OTFS frame selection directly from the repository root."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from otfs.core import OTFSConfig
from otfs.io import dump_json, load_json, select_dataset


DATASET_LAMBDAS = {"videomme": 0.5, "longvideobench": 0.005, "mlvu": 0.005}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OTFS frame selection")
    parser.add_argument("--input", required=True, help="Evidence JSON")
    parser.add_argument("--output", required=True, help="Detailed output JSON")
    parser.add_argument("--dataset", choices=sorted(DATASET_LAMBDAS), required=True)
    parser.add_argument("--budget", type=int, required=True, help="Frame budget (32 or 64 in the paper)")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--source-prior", type=float, nargs=3, default=(0.5, 0.4, 0.1))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--coverage-lambda", type=float, default=None)
    parser.add_argument("--coverage-gap", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage_lambda = (
        DATASET_LAMBDAS[args.dataset]
        if args.coverage_lambda is None
        else args.coverage_lambda
    )
    config = OTFSConfig(
        budget=args.budget,
        epsilon=args.epsilon,
        gamma=args.gamma,
        source_prior=tuple(args.source_prior),
        iterations=args.iterations,
        coverage_lambda=coverage_lambda,
        coverage_gap=args.coverage_gap,
    )
    data = load_json(args.input)
    if not isinstance(data, list):
        raise TypeError("input JSON must contain a list of question items")

    results = select_dataset(data, config)
    dump_json(results, args.output)
    output_path = Path(args.output)
    simple_path = output_path.with_name(output_path.stem + "_simple.json")
    dump_json([item["selected_frames"] for item in results], simple_path)

    print(f"Selected frames for {len(results)} items")
    print(f"Configuration: {asdict(config)}")
    print(f"Detailed output: {output_path}")
    print(f"Simple output: {simple_path}")


if __name__ == "__main__":
    main()
