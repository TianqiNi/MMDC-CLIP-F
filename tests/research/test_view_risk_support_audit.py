import json

import pytest
import torch

from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.support_audit import (
    audit_cells, evaluate_removals, summarize_cells,
)
from mmdc_clip_f.research.view_risk.targets import build_intervention_targets


def test_removal_audit_matches_targets_and_preserves_soft_changes():
    generator = torch.Generator().manual_seed(9)
    logits = {v: torch.randn(40, 4, generator=generator) * 3 for v in CANONICAL_VIEWS}
    labels = torch.arange(40) % 4
    result = evaluate_removals(logits, labels)
    expected = build_intervention_targets(logits, labels, CANONICAL_VIEWS)
    assert torch.equal(result['effects'], expected.omission_effects)
    assert torch.equal(result['error'], expected.observed_error)
    same = result['prediction'][:, None] == result['child_prediction']
    assert (result['effects'][same] == 0).all()
    assert (result['probability_tv'][same] > 0).any()
    assert ((result['effects'] == 0) & ~same).any()  # Wrong-to-wrong is unchanged.
    assert set(result['effects'].unique().tolist()) == {-1, 0, 1}


def test_summary_counts_patients_not_removals_and_reports_degenerate_ranking():
    logits = {v: torch.tensor([[3., 0., 0., 0.]] * 3) for v in CANONICAL_VIEWS}
    labels = torch.tensor([1, 1, 1])
    result = evaluate_removals(logits, labels)
    report = summarize_cells({'clean': result}, labels, ['private-p1', 'private-p1', 'private-p2'])
    cell = report['cells']['clean']
    assert cell['support']['removals'] == 12
    assert cell['support']['unchanged'] == 12
    assert cell['support']['unique_patients_any_unchanged'] == 2
    assert cell['support']['unique_patients_any_repair'] == 0
    assert cell['all_removals_unchanged_and_parent_wrong'] == 3
    assert cell['rankings']['msp']['error_auroc'] is None
    assert cell['rankings']['msp']['aurc'] == pytest.approx(1)
    assert 'private-p' not in json.dumps(report)
    with pytest.raises(ValueError, match='patient'):
        summarize_cells({'clean': result}, labels, ['one'])


def test_audit_cells_preserve_fitting_family_holdout():
    cells = audit_cells()
    assert len(cells) == 21
    assert cells[0] == ('clean', 0, ())
    assert len({name for name, _, _ in cells}) == 21
    for name, variant, stressed in cells[1:]:
        assert name.split('/')[0] in {'gaussian_noise', 'gaussian_blur'}
        assert name.split('/')[1] in {'mild', 'moderate'}
        assert 1 <= variant <= 4
        assert len(stressed) in (1, 4)


def test_preparation_exact_clean_transform_and_role_rejection(tmp_path):
    from PIL import Image

    from mmdc_clip_f.data import build_transforms
    from mmdc_clip_f.research.view_risk.roles import Role, ViewReference
    from mmdc_clip_f.research.view_risk.support_audit import prepare_record

    image = Image.new('RGB', (30, 40), (100, 80, 60))
    for view in CANONICAL_VIEWS:
        image.save(tmp_path / f'{view}.png')
    fields = dict(dataset_namespace='RSNA-SMBC', exam_key='exam', patient_key='patient',
                  density=2, source_manifest='fixture', role=Role.CONFIDENCE_FIT,
                  views={v: ViewReference(v, f'{v}.png') for v in CANONICAL_VIEWS})
    before = torch.get_rng_state().clone()
    result = prepare_record(fields, tmp_path, 224, 4242)
    assert torch.equal(torch.get_rng_state(), before)
    _, transform = build_transforms(224, None)
    assert torch.equal(result[0], torch.stack([transform(image)] * 4))
    assert torch.equal(result, prepare_record(fields, tmp_path, 224, 4242))
    # Identical inputs share the common-mode noise realization across views.
    assert torch.equal(result[1, 0], result[1, 3])
    fields['role'] = Role.PILOT
    fields['views'] = {v: ViewReference(v, f'missing-{v}.png') for v in CANONICAL_VIEWS}
    with pytest.raises(PermissionError):
        prepare_record(fields, tmp_path, 224, 4242)
