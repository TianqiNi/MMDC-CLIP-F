from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mmdc_clip_f.research.view_risk.artifacts import (
    DSControlRows,
    ScalarControlRows,
    create_raw_control_artifact,
    fit_ds_control_artifact_with_role_access,
    fit_temperature_control_artifact_with_role_access,
    load_control_artifact,
)
from mmdc_clip_f.research.view_risk.baselines import DSBaselineFeatures
from mmdc_clip_f.research.view_risk.fusion import RSNA_FUSION_PAIRS
from mmdc_clip_f.research.view_risk.training import (
    ClassifierProvenance,
    ResearchRunConfig,
    audit_software_fixture,
)
from mmdc_clip_f.research.view_risk.roles import Role
from tests.research.test_view_risk_training import _manifest


def _rows(manifest, scores):
    return ScalarControlRows(
        tuple(record.exam_key for record in manifest.records), scores
    )


def _fresh_selected(tune_manifest):
    fit = _manifest(Role.CLASSIFIER_FIT, 1)
    return ClassifierProvenance.fresh_selected(
        ClassifierProvenance.synthetic_injected("vit_b_32", "a" * 64),
        checkpoint_sha256="c" * 64,
        classifier_fit_manifest_sha256=fit.manifest_sha256,
        classifier_fit_update_count=1,
        tune_selection_sha256="e" * 64,
        tune_manifest_sha256=tune_manifest.manifest_sha256,
        readiness=audit_software_fixture(fit),
    )


def test_temperature_artifact_authorizes_before_reader_and_binds_fitted_state(tmp_path) -> None:
    config = ResearchRunConfig.default("RSNA")
    forbidden = _manifest(Role.CONFIDENCE_FIT)
    opened = False

    def reader(_records):
        nonlocal opened
        opened = True
        return _rows(forbidden, torch.tensor([[4.0, 1.0, 0.0, -1.0]]))

    with pytest.raises(PermissionError):
        fit_temperature_control_artifact_with_role_access(
            tmp_path / "forbidden.json",
            config=config,
            classifier=_fresh_selected(_manifest(Role.TUNE, 1)),
            seed=42,
            tune_manifest=forbidden,
            row_reader=reader,
        )
    assert opened is False

    tune = replace(_manifest(Role.TUNE, 1), records=tuple(_manifest(Role.TUNE, 1).records))
    classifier = _fresh_selected(tune)
    artifact = fit_temperature_control_artifact_with_role_access(
        tmp_path / "temperature.json",
        config=config,
        classifier=classifier,
        seed=42,
        tune_manifest=tune,
        row_reader=lambda _records: _rows(
            tune, torch.tensor([[4.0, 1.0, 0.0, -1.0]])
        ),
    )
    assert artifact.method == "temperature_scaled_msp"
    assert artifact.parameter_count == 1
    assert artifact.update_count == 1
    assert artifact.selection_trials == 1
    assert artifact.exposure_by_role == (("tune", 1),)
    assert artifact.scaling == "tune_fitted_temperature"
    assert load_control_artifact(artifact.path).sha256 == artifact.sha256


def test_raw_and_ds_artifacts_record_method_specific_fitting_rules(tmp_path) -> None:
    config = ResearchRunConfig.default("RSNA")
    tune = _manifest(Role.TUNE, 1)
    classifier = _fresh_selected(tune)
    raw = create_raw_control_artifact(
        tmp_path / "raw.json",
        method="msp",
        config=config,
        classifier=classifier,
        seed=42,
    )
    assert raw.update_count == raw.selection_trials == raw.parameter_count == 0
    assert raw.scaling == "none_declared_raw_ranking"

    confidence = _manifest(Role.CONFIDENCE_FIT, 1)
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    valid = torch.ones_like(values, dtype=torch.bool)

    def ds_rows(manifest):
        return DSControlRows(
            tuple(record.exam_key for record in manifest.records),
            DSBaselineFeatures(
                values,
                valid,
                torch.ones(1, 4, dtype=torch.bool),
                torch.ones(1, 3, dtype=torch.bool),
                RSNA_FUSION_PAIRS,
            ),
            torch.tensor([1]),
        )

    fitted = fit_ds_control_artifact_with_role_access(
        tmp_path / "ds.json",
        config=config,
        classifier=classifier,
        seed=42,
        confidence_manifest=confidence,
        tune_manifest=tune,
        confidence_reader=lambda _records: ds_rows(confidence),
        tune_reader=lambda _records: ds_rows(tune),
    )
    assert fitted.method == "ds_logistic"
    assert fitted.selection_trials == len(config.search_table.ds_regularizations)
    assert fitted.parameter_count == 17
    assert fitted.exposure_by_role == (("confidence_fit", 1), ("tune", 1))
    assert fitted.scaling == "confidence_fit_fitted_feature_scaler"
