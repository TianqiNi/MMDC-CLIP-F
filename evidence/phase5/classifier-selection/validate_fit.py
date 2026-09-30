"""Validate completed real fit artifacts; emit only aggregate evidence."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.production import (
    load_classifier_fit_artifact, load_tensor_checkpoint_state,
)
from mmdc_clip_f.research.view_risk.training import load_research_run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    root = Path(args.run_directory)
    config = load_research_run_config(args.config)
    execution = json.loads((root / "execution-through-epoch-050-parallel.json").read_text())
    assert execution["exit_code"] == 0
    fit = load_classifier_fit_artifact(root / "classifier-fit-epoch-050.json",
                                       expected_config=config)
    assert fit.result.completed_epoch == 50
    assert fit.result.update_count == 53500 and fit.result.exposed_record_count == 3209
    assert fit.readiness.patient_ready and not fit.result.software_only
    assert [e for e, _ in fit.checkpoint_paths] == list(range(1, 51))
    finite_tensors = 0
    for _, path in fit.checkpoint_paths:
        state = load_tensor_checkpoint_state(path)
        for value in state.values():
            if value.is_floating_point():
                assert torch.isfinite(value).all()
                finite_tensors += 1
    metadata = json.loads((root / "classifier-resume.json").read_text())
    assert sha256_file(root / "classifier-resume.pt") == metadata["checkpoint_sha256"]
    resume = torch.load(root / "classifier-resume.pt", map_location="cpu", weights_only=True)
    assert resume["binding"] == metadata["binding"] == fit.binding.to_dict()
    assert resume["completed_epoch"] == 50 and resume["update_count"] == 53500
    assert resume["schedule_state"]["next_epoch"] == 51
    assert set(state) == set(resume["model_state"])
    assert all(torch.equal(v, resume["model_state"][k]) for k, v in state.items())
    groups = resume["optimizer_state"]["param_groups"]
    assert len(groups) == 1 and groups[0]["lr"] == 1e-7 and groups[0]["weight_decay"] == 1e-5
    optimizer_tensors = 0
    for item in resume["optimizer_state"]["state"].values():
        for value in item.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert torch.isfinite(value).all()
                optimizer_tensors += 1
    result = {
        "status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_mode": "orchestrator; no subagents", "completed_epoch": 50,
        "update_count": 53500, "classifier_fit_patients": 3209,
        "fit_artifact_sha256": fit.sha256, "config_sha256": config.sha256,
        "classifier_schedule_sha256": config.classifier_schedule.sha256,
        "checkpoint_count": 50, "finite_model_tensors_across_checkpoints": finite_tensors,
        "latest_model_resume_equal": True, "latest_model_tensor_count": len(state),
        "finite_optimizer_tensors": optimizer_tensors,
        "resume_sha256": metadata["checkpoint_sha256"],
        "next_epoch": 51, "rng_components": sorted(resume["rng_state"]),
        "parallel_run_seconds": execution["elapsed_seconds"],
        "validation_seconds": time.perf_counter() - start,
        "selection_or_test_outcomes_accessed": False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
