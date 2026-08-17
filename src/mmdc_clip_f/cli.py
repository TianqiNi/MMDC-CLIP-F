"""Command-line entry point for the supported MMDC-CLIP-F core workflows."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__


def _common_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Experiment YAML configuration")
    parser.add_argument("--data-root", help="Override dataset.image_root")
    parser.add_argument(
        "--manifest-dir",
        help="Directory containing train/valid/test manifests with configured filenames",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmdc-clip-f",
        description="Reproducible core workflows for MMDC-CLIP-F",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-data", help="Validate manifests, files, and split isolation"
    )
    _common_data_options(validate)
    validate.add_argument(
        "--no-check-files",
        action="store_true",
        help="Validate manifests without walking image paths",
    )

    train = subparsers.add_parser("train-classifier", help="Train the four-view Stage-1 classifier")
    _common_data_options(train)
    train.add_argument("--output-root", help="Override output.root")
    train.add_argument("--run-name", help="Explicit unique run directory name")
    train.add_argument("--device", help="Torch device, for example cuda:0 or cpu")

    # The confidence module is optional at parser-construction time so the
    # classifier remains inspectable in minimal environments.
    try:
        from .confidence.cli import add_confidence_subparsers

        add_confidence_subparsers(subparsers)
    except ImportError:
        pass
    return parser


def _json_print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .config import load_config

    if args.command == "validate-data":
        from .classifier import validate_data

        config = load_config(args.config)
        report = validate_data(
            config,
            data_root=args.data_root,
            manifest_dir=args.manifest_dir,
            check_files=not args.no_check_files,
        )
        _json_print(report)
        return 0 if report["valid"] else 2

    if args.command == "train-classifier":
        from .classifier import train_classifier

        config = load_config(args.config)
        run_dir = train_classifier(
            config,
            data_root=args.data_root,
            manifest_dir=args.manifest_dir,
            output_root=args.output_root,
            run_name=args.run_name,
            device_name=args.device,
        )
        print(run_dir)
        return 0

    try:
        from .confidence.cli import run_confidence_command

        handled = run_confidence_command(args)
        if handled is not None:
            return int(handled)
    except ImportError:
        pass
    raise RuntimeError(f"Command handler is unavailable: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
