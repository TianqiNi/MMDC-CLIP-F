"""CLI wiring for TCP construction and MV-ACN training."""

from __future__ import annotations

import argparse
import json


_COMMANDS = {
    "build-tcp",
    "train-confidence",
}


def _stage1_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Confidence YAML configuration")
    parser.add_argument("--stage1-config", required=True, help="Matching Stage-1 YAML")
    parser.add_argument("--stage1-run-dir", required=True, help="Completed Stage-1 run")
    parser.add_argument("--data-root", help="Override the authorized local image root")
    parser.add_argument("--manifest-dir", help="Override the local split-manifest directory")
    parser.add_argument("--device", help="Torch device, for example cuda:0 or cpu")


def add_confidence_subparsers(subparsers) -> None:
    build = subparsers.add_parser(
        "build-tcp", help="Build strict TCP artifacts from a verified Stage-1 run"
    )
    _stage1_options(build)
    build.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        help="Defaults to train plus the configured checkpoint-selection split",
    )
    build.add_argument("--output-root", help="Override confidence stage1.tcp_root")

    train = subparsers.add_parser(
        "train-confidence", help="Train the head-only MV-ACN confidence model"
    )
    _stage1_options(train)
    train.add_argument("--tcp-root", required=True, help="Directory containing split TCP artifacts")
    train.add_argument("--output-root", help="Override confidence output.root")
    train.add_argument("--run-name", help="Explicit unique confidence run-directory name")
    train.add_argument(
        "--allow-test-selection",
        action="store_true",
        help="Explicitly reproduce archived test-informed checkpoint selection",
    )


def run_confidence_command(args: argparse.Namespace) -> int | None:
    if args.command not in _COMMANDS:
        return None

    from ..config import load_config
    from .training import (
        build_tcp_artifacts,
        load_confidence_config,
        train_confidence,
    )

    confidence_config = load_confidence_config(args.config)
    stage1_config = load_config(args.stage1_config)
    if args.command == "build-tcp":
        splits = args.splits or tuple(
            dict.fromkeys(("train", str(confidence_config["selection"]["split"])))
        )
        result = build_tcp_artifacts(
            confidence_config,
            stage1_config,
            args.stage1_run_dir,
            splits=splits,
            data_root=args.data_root,
            manifest_dir=args.manifest_dir,
            output_root=args.output_root,
            device_name=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train-confidence":
        run_dir = train_confidence(
            confidence_config,
            stage1_config,
            args.stage1_run_dir,
            args.tcp_root,
            data_root=args.data_root,
            manifest_dir=args.manifest_dir,
            output_root=args.output_root,
            run_name=args.run_name,
            device_name=args.device,
            allow_test_selection=args.allow_test_selection,
        )
        print(run_dir)
        return 0

    raise RuntimeError(f"unhandled confidence command: {args.command}")
