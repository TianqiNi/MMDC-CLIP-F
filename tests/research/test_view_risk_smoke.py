from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
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


def test_smoke_scores_each_tree_method_and_panel_through_accepted_metrics(
    tmp_path, monkeypatch
) -> None:
    target_trees: Counter[object] = Counter()
    learned_score_trees: Counter[tuple[str, object]] = Counter()
    analytic_score_methods: Counter[str] = Counter()
    analytic_score_trees: Counter[tuple[str, object]] = Counter()
    score_tensor_trees: dict[int, object] = {}
    confidence_results: list[dict[str, object]] = []
    effect_results: list[dict[str, object]] = []

    original_targets = smoke_module.checked_intervention_targets
    original_learned = smoke_module.confidence_bundle_scores
    original_analytic = smoke_module.analytic_control_confidence
    original_confidence_metrics = smoke_module.confidence_panel_metrics
    original_effect_metrics = smoke_module.effect_metrics

    def observe_targets(features, cached_targets):
        target_trees[features.fusion_pairs] += features.batch_size
        score_tensor_trees[id(features.scores)] = features.fusion_pairs
        return original_targets(features, cached_targets)

    def observe_learned(model, method, bundle):
        learned_score_trees[(method, bundle.features.fusion_pairs)] += (
            bundle.features.batch_size
        )
        return original_learned(model, method, bundle)

    def observe_analytic(artifact, **kwargs):
        analytic_score_methods[artifact.method] += 1
        if kwargs.get("scores") is not None:
            tree = score_tensor_trees[id(kwargs["scores"])]
        else:
            tree = kwargs["ds_features"].fusion_pairs
        analytic_score_trees[(artifact.method, tree)] += 1
        return original_analytic(artifact, **kwargs)

    def observe_confidence_metrics(correct, confidence, *, confidence_kind):
        result = original_confidence_metrics(
            correct, confidence, confidence_kind=confidence_kind
        )
        confidence_results.append(asdict(result))
        return result

    def observe_effect_metrics(
        effect_targets, reported_probabilities, valid_slots, *, stressed
    ):
        result = original_effect_metrics(
            effect_targets, reported_probabilities, valid_slots, stressed=stressed
        )
        effect_results.append(asdict(result))
        return result

    monkeypatch.setattr(smoke_module, "checked_intervention_targets", observe_targets)
    monkeypatch.setattr(smoke_module, "confidence_bundle_scores", observe_learned)
    monkeypatch.setattr(smoke_module, "analytic_control_confidence", observe_analytic)
    monkeypatch.setattr(
        smoke_module, "confidence_panel_metrics", observe_confidence_metrics
    )
    monkeypatch.setattr(smoke_module, "effect_metrics", observe_effect_metrics)

    report = run_synthetic_smoke(
        output_dir=tmp_path / "behavioral-smoke",
        device="cpu",
        warmup=0,
        repeats=1,
    )

    expected_panels = {
        "clean_four_view": 2,
        "clean_masks": 28,
        "permitted_single": 2,
        "held_out_single": 2,
        "common_mode": 2,
    }
    expected_methods = {
        "candidate": "probability",
        "same_input_mlp": "probability",
        "msp": "ranking",
        "ds_logistic": "probability",
    }
    matrix = report["scientific_outputs"]["coverage_matrix"]
    reported_confidence_results = []
    reported_effect_results = []
    for method, confidence_kind in expected_methods.items():
        assert matrix[method]["training_scope"]
        for tree in ("RSNA", "DDSM"):
            scoring = matrix[method]["scoring"][tree]
            assert set(scoring) == set(expected_panels)
            for panel, expected_count in expected_panels.items():
                result = scoring[panel]
                assert result["sample_count"] == expected_count
                assert result["checked_target_count"] == expected_count
                assert result["prediction_consistent_count"] == expected_count
                assert result["confidence_kind"] == confidence_kind
                assert result["confidence_metrics"]["n_exams"] == expected_count
                assert result["confidence_metrics"]["confidence_kind"] == confidence_kind
                if confidence_kind == "probability":
                    assert result["confidence_metrics"]["brier"] is not None
                else:
                    assert result["confidence_metrics"]["brier"] is None
                reported_confidence_results.append(result["confidence_metrics"])
                if method == "candidate":
                    reported_effect_results.append(result["effect_metrics"])
                else:
                    assert result["effect_metrics"] == "not_applicable_no_effect_output"
            assert scoring["clean_masks"]["aurc_panel"]["cell_count"] == 14

    assert target_trees.total() == 72
    assert sorted(target_trees.values()) == [36, 36]
    assert all(
        learned_score_trees[(method, tree)] >= 36
        for method in ("candidate", "same_input_mlp")
        for tree in target_trees
    )
    assert analytic_score_methods["msp"] >= 72
    assert analytic_score_methods["ds_logistic"] >= 72
    assert all(
        analytic_score_trees[(method, tree)] >= 36
        for method in ("msp", "ds_logistic")
        for tree in target_trees
    )
    assert Counter(
        json.dumps(value, sort_keys=True) for value in reported_confidence_results
    ) == Counter(json.dumps(value, sort_keys=True) for value in confidence_results)
    assert Counter(
        json.dumps(value, sort_keys=True) for value in reported_effect_results
    ) == Counter(json.dumps(value, sort_keys=True) for value in effect_results)


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
