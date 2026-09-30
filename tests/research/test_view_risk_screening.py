"""Scientific invariants for the bounded, tune-only screening experiment."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import save_file

from mmdc_clip_f.research.view_risk.head import (
    RelationAwareConfidenceHead, RelationAwareHeadConfig, compute_view_risk_loss,
)
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.screening import (
    matched_screening_model, pack_screening_batch, slice_screening_batch,
    summarize_screening_scores, choose_screening_epoch,
    validate_screening_labels, realize_scheduled_parent,
)
from mmdc_clip_f.research.view_risk.perturbations import sample_training_spec, realize_parent
from tests.research.test_view_risk_head import _raw, _targets
from tests.research.test_view_risk_features import FakeCLIP
from tests.research.test_view_risk_training import _manifest
from mmdc_clip_f.model import MultiViewCLIPClassifier
from mmdc_clip_f.backbones import PROMPTS
from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.roles import Role, RoleManifest
from mmdc_clip_f.research.view_risk.fusion import RSNA_FUSION_PAIRS
from mmdc_clip_f.research.view_risk.training import ResearchRunConfig


def test_mixed_masks_keep_original_order_and_parent_normalized_loss_and_gradient():
    groups = []
    for mask in [CANONICAL_VIEWS, ("L_CC",), ("L_MLO", "R_MLO")]:
        raw, logits = _raw(mask)
        groups.append((raw, _targets(logits, mask)))
    packed = pack_screening_batch(groups, [[0, 3], [1, 4], [2, 5]])
    assert packed.raw.observed_mask.sum(1).tolist() == [4, 1, 2, 4, 1, 2]
    assert packed.targets.valid_removal_mask.sum(1).tolist() == [4, 0, 2, 4, 0, 2]
    torch.manual_seed(21)
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32", dropout=0))
    model.eval()
    mixed = compute_view_risk_loss(model(packed.raw), packed.targets).total
    mixed.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    individual = torch.stack([compute_view_risk_loss(model(r), t).total for r, t in groups]).mean()
    individual.backward()
    torch.testing.assert_close(mixed, individual)
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[k], atol=1e-6, rtol=1e-4)
    one = slice_screening_batch(packed, [1], device="cpu")
    assert one.raw.batch_size == 1 and one.targets.valid_removal_mask.sum() == 0


def test_screening_rejects_duplicate_rows_and_stale_target_predictions():
    raw, logits = _raw()
    target = _targets(logits, ("L_CC", "L_MLO", "R_CC"))
    with pytest.raises(ValueError, match="permutation"):
        pack_screening_batch([(raw, target)], [[0, 0]])
    stale = replace(target, observed_prediction=(target.observed_prediction + 1) % 4)
    with pytest.raises(ValueError, match="prediction"):
        pack_screening_batch([(raw, stale)], [[0, 1]])


def test_controls_have_identical_shared_initialization_and_seed_rng_after_build():
    models = [matched_screening_model(m, 42, "vit_b_32") for m in
              ("candidate", "no_effect_supervision", "magnitude_effects")]
    reference = models[0].state_dict()
    for model in models[1:]:
        for k, x in model.state_dict().items():
            if x.shape == reference[k].shape:
                assert torch.equal(x, reference[k])
    assert models[2].config.auxiliary_task == "magnitude"
    with pytest.raises(ValueError, match="screening"):
        matched_screening_model("same_input_mlp", 42, "vit_b_32")


def test_selection_uses_equal_cell_weighting_and_earliest_tie_and_keeps_clean_separate():
    error = np.tile([0, 0, 1, 1], (17, 1))
    confidence = np.tile([.9, .8, .7, .6], (17, 1))
    confidence[0] = confidence[0, ::-1]
    metrics = summarize_screening_scores(error, confidence)
    assert metrics["clean_aurc"] > metrics["balanced_aurc"]
    assert metrics["balanced_aurc"] == pytest.approx(np.mean([x["aurc"] for x in metrics["cells"][1:]]))
    assert choose_screening_epoch([{ "epoch": 2, **metrics}, {"epoch": 1, **metrics}])["epoch"] == 1
    with pytest.raises(ValueError, match="17"):
        summarize_screening_scores(error[:2], confidence[:2])


def test_cache_targets_are_verified_against_current_labels_not_only_predictions():
    raw, logits = _raw(CANONICAL_VIEWS)
    batch = pack_screening_batch([(raw, _targets(logits, CANONICAL_VIEWS))], [[0, 1]])
    validate_screening_labels(batch, torch.tensor([0, 1]))
    corrupted = replace(batch, targets=replace(batch.targets, omission_effects=torch.ones(2, 4).long()))
    with pytest.raises(ValueError, match="effects"):
        validate_screening_labels(corrupted, torch.tensor([0, 1]))


def test_scheduled_realizations_replay_existing_common_and_single_view_api_exactly():
    images = {v: torch.linspace(0, 1, 3 * 12 * 12).reshape(3, 12, 12) for v in CANONICAL_VIEWS}
    modes = set()
    for epoch in range(1, 100):
        draw = sample_training_spec("private-example", f"epoch/{epoch}", 42, operation="confidence-fit")
        actual = realize_scheduled_parent(images, draw)
        if draw.stress_mode == "common":
            reference = realize_parent(images, draw.observed_views, common_mode=draw.perturbation)
        else:
            reference = realize_parent(images, draw.observed_views, perturbations={v: draw.perturbation for v in draw.stressed_views})
        assert actual.observed_views == draw.observed_views
        for v in actual.observed_views:
            assert torch.equal(actual.images[v], reference.images[v])
        modes.add(draw.stress_mode)
    assert modes == {None, "single", "common"}


def test_private_screening_runner_resumes_saved_epoch_and_finishes_all_nine_runs(tmp_path, monkeypatch):
    """Exercise real caching, role-bound optimizer, callbacks and exact resume on toy pixels."""
    path = Path(__file__).resolve().parents[2] / 'evidence/phase5/signed-screening/run_screening.py'
    spec = importlib.util.spec_from_file_location('screening_runner_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = ResearchRunConfig.default('RSNA')
    fit = _manifest(Role.CONFIDENCE_FIT)
    original_tune = _manifest(Role.TUNE)
    tune = RoleManifest(dataset_namespace=original_tune.dataset_namespace,
        source_hashes=original_tune.source_hashes, patient_mapping=original_tune.patient_mapping,
        records=[replace(r, exam_key='tune-' + r.exam_key, patient_key='tune-' + r.patient_key)
                 for r in original_tune.records])
    classifier = MultiViewCLIPClassifier(FakeCLIP(768), CANONICAL_VIEWS, RSNA_FUSION_PAIRS)
    checkpoint = tmp_path / 'classifier.safetensors'
    save_file({k: v.contiguous() for k, v in classifier.state_dict().items()}, str(checkpoint))
    selected = SimpleNamespace(sha256='a' * 64, selected_checkpoint_path=checkpoint,
        classifier=SimpleNamespace(checkpoint_sha256=sha256_file(checkpoint), image_size=12, prompts=PROMPTS),
        initialization=None)
    monkeypatch.setattr(runner, 'load_research_run_config', lambda path: config)
    monkeypatch.setattr(runner, '_manifest_binding', lambda path: None)
    monkeypatch.setattr(runner, 'load_role_manifest_for_operation', lambda path, **kw:
                        fit if 'confidence-fit' in str(path) else tune)
    monkeypatch.setattr(runner, 'load_selected_classifier_artifact', lambda *a, **kw: selected)
    def reload(*a, **kw):
        model = MultiViewCLIPClassifier(FakeCLIP(768), CANONICAL_VIEWS, RSNA_FUSION_PAIRS)
        return model, torch.arange(8).reshape(4, 2)
    monkeypatch.setattr(runner, 'reload_verified_public_classifier', reload)
    monkeypatch.setattr(runner, 'decode_record', lambda *a:
                        torch.stack([torch.full((3, 12, 12), .2 + i * .1) for i in range(4)]))
    @contextmanager
    def pool(workers):
        with ThreadPoolExecutor(max_workers=2) as executor:
            yield executor
    monkeypatch.setattr(runner, 'image_worker_pool', pool)
    plan = json.loads((path.parent / 'plan.json').read_text())
    plan.update(config_sha256=config.sha256, expected_fit_exams=4, expected_tune_exams=4,
                confidence_fit_manifest_sha256=fit.manifest_sha256, tune_manifest_sha256=tune.manifest_sha256,
                selected_artifact_sha256=selected.sha256, classifier_checkpoint_sha256=sha256_file(checkpoint))
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    private, summary = tmp_path / 'private', tmp_path / 'summary.json'
    arguments = ['screen', '--config', 'fixture', '--plan', str(plan_path), '--selected', 'fixture',
        '--role-root', str(tmp_path), '--image-root', 'fixture', '--private-output', str(private),
        '--summary-output', str(summary), '--device', 'cpu', '--max-seconds', '0']
    monkeypatch.setattr(sys, 'argv', arguments)
    original_write = runner.write_json
    def interrupted_callback(path, value, **kwargs):
        if path.name == 'history.json':
            raise OSError('simulated interruption after epoch checkpoint, before history')
        return original_write(path, value, **kwargs)
    monkeypatch.setattr(runner, 'write_json', interrupted_callback)
    with pytest.raises(SystemExit):
        runner.main()
    assert json.loads((private / 'execution.json').read_text())['status'] == 'FAIL'
    saved = torch.load(private / 'candidate-seed-42/resume.pt', map_location='cpu', weights_only=True)
    assert saved['completed_epoch'] == 1
    monkeypatch.setattr(runner, 'write_json', original_write)
    with pytest.raises(SystemExit):
        runner.main()
    assert json.loads((private / 'execution.json').read_text())['status'] == 'STOPPED'
    arguments[-1] = '3600'
    monkeypatch.setattr(sys, 'argv', arguments)
    runner.main()
    result = json.loads(summary.read_text())
    assert len(result['results']) == 9
    assert result['provenance']['pilot_or_test_outcomes_accessed'] is False
    for run in result['results'].values():
        assert run['training']['completed_epoch'] == 20
        assert run['training']['update_count'] == 20
        assert [x['epoch'] for x in run['history']] == list(range(1, 21))
