"""Authorized classifier-fit pixels/RNG and loader timing; emits aggregates only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mmdc_clip_f.cli import build_parser
from mmdc_clip_f.research.view_risk.cli import _manifest_binding
from mmdc_clip_f.research.view_risk.production import role_image_batch_loader
from mmdc_clip_f.research.view_risk.roles import Operation, Role
from mmdc_clip_f.research.view_risk.training import (
    load_research_run_config,
    load_role_manifest_for_operation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patients", type=int, default=60)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4, 8, 12])
    args = parser.parse_args()
    command = json.loads(Path(args.command_file).read_text())
    argv = command["argv"]
    config_args = build_parser().parse_args(argv[argv.index("view-risk-fit-classifier"):])
    config = load_research_run_config(config_args.config)
    manifest = load_role_manifest_for_operation(
        config_args.manifest, expected=_manifest_binding(config_args.manifest_binding),
        private_root=config_args.private_root, operation=Operation.CLASSIFIER_FITTING,
        role=Role.CLASSIFIER_FIT,
    )
    records = manifest.records[:args.patients]
    output = {"role": "classifier_fit", "patients": len(records), "runs": []}
    reference = None
    reference_rng = None
    for workers in args.workers:
        torch.manual_seed(42)
        start = time.perf_counter()
        arrivals = []
        hashes = []
        for batch in role_image_batch_loader(
            records, image_root=config_args.image_root, image_size=224,
            optimizer=config.classifier_schedule.optimizer,
            augmentation=config.classifier_schedule.augmentation,
            epoch=16, seed=42, training=True, image_workers=workers, prefetch_batches=2,
        ):
            arrivals.append(time.perf_counter() - start)
            digest = hashlib.sha256()
            for value in batch.payload.views.values():
                digest.update(value.numpy().tobytes())
            digest.update(batch.payload.labels.numpy().tobytes())
            hashes.append(digest.hexdigest())
        elapsed = time.perf_counter() - start
        rng = torch.get_rng_state().clone()
        if reference is None:
            reference, reference_rng = hashes, rng
        equal = hashes == reference and torch.equal(rng, reference_rng)
        result = {"workers": workers, "total_seconds": elapsed,
                  "first_batch_seconds": arrivals[0],
                  "steady_seconds_per_batch": (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1),
                  "batches": len(arrivals), "serial_pixels_labels_rng_equal": equal}
        output["runs"].append(result)
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not equal:
            raise RuntimeError("real-image serial equivalence failed")


if __name__ == "__main__":
    main()
