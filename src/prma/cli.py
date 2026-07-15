"""Command-line interface for reviewed PRMA experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prma.experiment import load_experiment_config, run_capacity_experiment, write_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prma", description="PRMA research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="run a sequential capacity experiment")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "experiment":
        config = load_experiment_config(args.config)
        result = run_capacity_experiment(config)
        write_result(result, args.output)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        print(f"result: {args.output}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
