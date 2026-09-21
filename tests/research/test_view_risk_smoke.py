from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmdc_clip_f.cli import main
from mmdc_clip_f.research.view_risk import smoke as smoke_module
from mmdc_clip_f.research.view_risk.smoke import (
    measure_repeated_cost,
    resource_preflight,
    run_synthetic_smoke,
)


def test_resource_preflight_is_read_only_and_does_not_promote_missing_resources() -> None:
    report = resource_preflight(device="cpu")

    assert report["mode"] == "software_resource_preflight"
    assert report["network_attempted"] is False
    assert report["downloads_or_installs_attempted"] is False
    assert report["requested_device"]["status"] == "available"
    assert report["private_data"] == "unavailable_not_audited"
    assert report["patient_mapping"] == "unavailable_not_audited"
    assert report["locked_outcomes_opened"] is False
    assert report["real_patient_readiness"] == "unavailable_not_audited"
    assert all("path" not in key.lower() for key in report)


def test_repeated_cost_measurement_has_explicit_calls_units_and_cpu_memory_semantics() -> None:
    calls: list[str] = []
    times = iter((0, 1_000_000, 10_000_000, 13_000_000, 20_000_000, 22_000_000))

    result = measure_repeated_cost(
        "controlled_work",
        lambda: calls.append("work"),
        device="cpu",
        warmup=1,
        repeats=3,
        _clock_ns=lambda: next(times),
    )

    assert calls == ["work"] * 4
    assert result["measurement_status"] == "measured"
    assert result["unit"] == "milliseconds_per_call"
    assert result["warmup_calls"] == 1
    assert result["timed_calls"] == 3
    assert result["median"] == pytest.approx(2.0)
    assert result["minimum"] == pytest.approx(1.0)
    assert result["maximum"] == pytest.approx(3.0)
    assert result["synchronization"] == "not_applicable_cpu"
    assert result["memory"]["metric"] == "cpu_process_high_water_rss"
    assert result["memory"]["scope"] == "process_lifetime_high_water_not_operation_delta"


def test_public_cli_smoke_is_reproducible_and_non_promotable(tmp_path, capsys) -> None:
    first_output = tmp_path / "first-smoke"
    second_output = tmp_path / "second-smoke"

    assert main(
        [
            "view-risk-smoke",
            "--output-dir",
            str(first_output),
            "--warmup",
            "0",
            "--repeats",
            "2",
        ]
    ) == 0
    first = json.loads(capsys.readouterr().out)
    second = run_synthetic_smoke(
        output_dir=second_output,
        device="cpu",
        warmup=0,
        repeats=2,
    )

    assert first["schema_version"] == "view-risk-p4c-smoke-report/v1"
    assert first["mode"] == "synthetic_software_only"
    assert first["status"] == "software_smoke_passed"
    assert first["scientific_conclusions"] == {
        "empirical_benefit": "unavailable",
        "pilot_go_no_go": "unavailable",
        "real_backbone_efficiency": "unavailable_unmeasured",
        "real_patient_readiness": "unavailable_not_audited",
    }
    assert first["coverage"]["nonempty_mask_count"] == 15
    assert first["coverage"]["singleton_mask_count"] == 4
    assert set(first["coverage"]["fusion_trees_exercised"]) == {"RSNA", "DDSM"}
    assert set(first["coverage"]["signed_effect_support"]) == {"-1", "0", "+1"}
    assert all(value > 0 for value in first["coverage"]["signed_effect_support"].values())
    assert first["invariants"]["classifier_predictions_fixed_across_methods"] is True
    assert first["invariants"]["targets_regenerated_from_realized_parent"] is True
    assert first["invariants"]["singleton_removal_slots_invalid"] is True
    assert {"candidate", "same_input_mlp", "msp", "ds_logistic"}.issubset(
        first["methods"]["exercised"]
    )
    assert first["fit_reload"]["candidate"]["checkpoint_reloaded"] is True
    assert first["fit_reload"]["same_input_mlp"]["checkpoint_reloaded"] is True
    assert first["fit_reload"]["candidate"]["completed_epochs"] == 1
    assert first["fit_reload"]["candidate"]["configured_epochs"] == 20
    assert first["reproducibility"]["scientific_sha256"] == second["reproducibility"][
        "scientific_sha256"
    ]
    assert first["scientific_outputs"] == second["scientific_outputs"]
    assert first["costs"] != second["costs"]
    assert (first_output / "report.json").is_file()
    assert (second_output / "report.json").is_file()
    public_text = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in public_text
    assert "exam-" not in public_text
    assert "patient-" not in public_text


def test_smoke_refuses_existing_or_unignored_output_without_touching_user_files(
    tmp_path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must not already exist"):
        run_synthetic_smoke(output_dir=existing, warmup=0, repeats=1)
    assert sentinel.read_text(encoding="utf-8") == "preserve me"

    worktree = Path(__file__).resolve().parents[2]
    refused = worktree / "unignored-p4c-smoke-output"
    assert not refused.exists()
    with pytest.raises(ValueError, match="gitignore"):
        run_synthetic_smoke(output_dir=refused, warmup=0, repeats=1)
    assert not refused.exists()


def test_smoke_cleans_staging_resources_when_execution_fails(tmp_path, monkeypatch) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    output = tmp_path / "failed-smoke"

    def fail(workspace, **_kwargs):
        (workspace / "partial-artifact").write_text("partial", encoding="utf-8")
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(smoke_module, "_run_smoke_workspace", fail)
    with pytest.raises(RuntimeError, match="controlled failure"):
        run_synthetic_smoke(output_dir=output, warmup=0, repeats=1)

    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not tuple(tmp_path.glob(".failed-smoke.staging-*"))


def test_smoke_cleans_automatic_temporary_resources_on_success(monkeypatch) -> None:
    workspaces: list[Path] = []

    def succeed(workspace, **_kwargs):
        workspaces.append(workspace)
        (workspace / "temporary-artifact").write_text("temporary", encoding="utf-8")
        return {"artifact_summary": {}}

    monkeypatch.setattr(smoke_module, "_run_smoke_workspace", succeed)
    report = run_synthetic_smoke(warmup=0, repeats=1)

    assert report["artifacts_retained"] is False
    assert report["artifact_summary"]["retention"] == "temporary_resources_cleaned"
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


def test_smoke_does_not_clobber_destination_created_during_run(tmp_path, monkeypatch) -> None:
    output = tmp_path / "late-destination"

    def complete(workspace, **_kwargs):
        (workspace / "complete-artifact").write_text("complete", encoding="utf-8")
        output.mkdir()
        (output / "user-file").write_text("preserve", encoding="utf-8")
        return {"artifact_summary": {}}

    monkeypatch.setattr(smoke_module, "_run_smoke_workspace", complete)
    with pytest.raises(FileExistsError):
        run_synthetic_smoke(output_dir=output, warmup=0, repeats=1)

    assert (output / "user-file").read_text(encoding="utf-8") == "preserve"
    assert not tuple(tmp_path.glob(".late-destination.staging-*"))
