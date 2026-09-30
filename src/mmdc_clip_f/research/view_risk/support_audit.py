"""Descriptive, full-parent removal audit; no fitting or held-out selection."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from .features import LEGACY_IMAGE_MEAN, LEGACY_IMAGE_STD
from .fusion import fuse_view_logits
from .inputs import CANONICAL_VIEWS
from .metrics import confidence_panel_metrics
from .perturbations import FITTING_FAMILIES, FITTING_SEVERITIES, PerturbationSpec, realize_parent
from .roles import PrivateExamRecord, Role
from .targets import build_intervention_targets


VARIANTS = tuple((f, s) for f in FITTING_FAMILIES for s in FITTING_SEVERITIES)


def audit_cells():
    cells = [('clean', 0, ())]
    for index, (family, severity) in enumerate(VARIANTS, 1):
        for view in CANONICAL_VIEWS:
            cells.append((f'{family}/{severity}/{view}', index, (view,)))
        cells.append((f'{family}/{severity}/common', index, CANONICAL_VIEWS))
    return tuple(cells)


def prepare_record(fields, image_root, image_size, seed):
    """Decode once; realize stresses before normalization in a CPU worker.

    The same realized stress is reused between single-view and common-mode cells.
    Within a common-mode cell the existing perturbation API shares its noise field.
    """
    from .production import default_private_image_reader

    record = PrivateExamRecord(**fields)
    if record.role != Role.CONFIDENCE_FIT:
        raise PermissionError('support audit requires confidence-fit records')
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    normalize = transforms.Normalize(LEGACY_IMAGE_MEAN, LEGACY_IMAGE_STD)
    images = {v: transform(default_private_image_reader(record, v, image_root))
              for v in CANONICAL_VIEWS}
    variants = [{v: normalize(x) for v, x in images.items()}]
    for family, severity in VARIANTS:
        spec = PerturbationSpec.for_sample(
            family, severity, private_sample_key=record.exam_key,
            cell=f'support-audit/v1/{family}/{severity}', seed=seed,
        )
        parent = realize_parent(images, CANONICAL_VIEWS, common_mode=spec)
        variants.append({v: normalize(x) for v, x in parent.images.items()})
    return torch.stack([torch.stack([x[v] for v in CANONICAL_VIEWS]) for x in variants])


def evaluate_removals(logits_by_view, labels):
    """Return private per-exam tensors using the unchanged RSNA fusion tree."""
    with torch.no_grad():
        parent = fuse_view_logits(logits_by_view, CANONICAL_VIEWS)
        target = build_intervention_targets(logits_by_view, labels, CANONICAL_VIEWS)
        children = [fuse_view_logits(logits_by_view, tuple(v for v in CANONICAL_VIEWS if v != out))
                    for out in CANONICAL_VIEWS]
        child_scores = torch.stack([x.scores for x in children], dim=1)
        probability = parent.scores.softmax(-1)
        child_probability = child_scores.softmax(-1)
        tv = (probability[:, None] - child_probability).abs().sum(-1) / 2
        return {
            'parent_scores': parent.scores.cpu(),
            'prediction': target.observed_prediction.cpu(),
            'child_prediction': target.omission_predictions.cpu(),
            'error': target.observed_error.cpu(),
            'effects': target.omission_effects.cpu(),
            'probability_tv': tv.cpu(),
            'msp': probability.max(-1).values.cpu(),
            'msp_delta': (child_probability.max(-1).values
                          - probability.max(-1).values[:, None]).cpu(),
            'evidential_uncertainty_delta': torch.cat(
                [x.stats.uncertainty - parent.stats.uncertainty for x in children], dim=1,
            ).cpu(),
            'score_sensitivity': (parent.scores[:, None] - child_scores).abs().mean(-1)
                                .max(-1).values.cpu(),
        }


def _support(effects, patients):
    # Each row is one parent, each column a removed view. A patient is counted
    # once per effect category even when several views/cells produce that effect.
    result = {'parents': len(effects), 'removals': effects.size,
              'unique_patients': len(set(patients))}
    for value, name in [(-1, 'damage'), (0, 'unchanged'), (1, 'repair')]:
        mask = effects == value
        result[name] = int(mask.sum())
        result[f'unique_patients_any_{name}'] = len(
            {p for p, present in zip(patients, mask.any(axis=1)) if present}
        )
    return result


def _distribution(values):
    values = np.asarray(values).reshape(-1)
    if not len(values):
        return {'count': 0, 'mean': None, 'median': None, 'p90': None}
    return {'count': len(values), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'p90': float(np.quantile(values, .9))}


def _ranking(error, confidence):
    metrics = confidence_panel_metrics(1 - error, confidence, confidence_kind='ranking')
    return {'error_auroc': float(roc_auc_score(error, -confidence))
            if len(np.unique(error)) == 2 else None,
            'error_ap': metrics.error_positive_ap.value, 'aurc': metrics.aurc,
            'error_count': int(error.sum()), 'parents': len(error)}


def summarize_cells(cells, labels, patient_keys):
    """Emit aggregates only. Metrics are within-cell descriptive diagnostics."""
    y = np.asarray(labels)
    if len(patient_keys) != len(y) or not len(y):
        raise ValueError('patient keys must match nonempty labels')
    output = {'patients': len(set(patient_keys)), 'exams': len(y), 'cells': {}, 'strata': {}}
    strata = {'clean': [], 'single_view_stress': [], 'common_mode_stress': []}
    for name, tensors in cells.items():
        d = {k: np.asarray(v) for k, v in tensors.items()}
        if any(len(v) != len(y) for v in d.values()):
            raise ValueError('cell tensors must match labels and patient keys')
        effect = d['effects']
        if effect.shape != (len(y), 4) or not np.isin(effect, [-1, 0, 1]).all():
            raise ValueError('invalid full-parent effect matrix')
        same = d['prediction'][:, None] == d['child_prediction']
        if not (effect[same] == 0).all():
            raise ValueError('same prediction must have zero effect')
        tv_same = d['probability_tv'][same]
        cell = {
            'support': _support(effect, patient_keys),
            'accuracy': float(1 - d['error'].mean()),
            'prediction_flips': int((~same).sum()),
            'all_removals_unchanged_and_parent_wrong': int(
                ((effect == 0).all(1) & (d['error'] == 1)).sum()),
            'same_prediction_tv': _distribution(tv_same),
            'same_prediction_tv_gt_1e_6': int((tv_same > 1e-6).sum()),
            'same_prediction_tv_at_least_0_05': int((tv_same >= .05).sum()),
            'msp_delta': _distribution(d['msp_delta']),
            'evidential_uncertainty_delta': _distribution(d['evidential_uncertainty_delta']),
            'by_removed_view': {v: _support(effect[:, i:i+1], patient_keys)
                                for i, v in enumerate(CANONICAL_VIEWS)},
            'by_density': {str(c): _support(effect[y == c],
                            [p for p, selected in zip(patient_keys, y == c) if selected])
                           for c in range(4)},
            'rankings': {
                'msp': _ranking(d['error'], d['msp']),
                'score_sensitivity': _ranking(d['error'], -d['score_sensitivity']),
                'probability_tv': _ranking(d['error'], -d['probability_tv'].max(1)),
                'prediction_flip_fraction': _ranking(d['error'], -(~same).mean(1)),
            },
        }
        output['cells'][name] = cell
        stratum = 'clean' if name == 'clean' else (
            'common_mode_stress' if name.endswith('/common') else 'single_view_stress')
        strata[stratum].append(name)
    for stratum, names in strata.items():
        if not names:
            continue
        effects = np.concatenate([np.asarray(cells[n]['effects']) for n in names])
        output['strata'][stratum] = {
            'cells': len(names), 'support': _support(effects, list(patient_keys) * len(names)),
            'mean_cell_aurc': {method: float(np.mean([
                output['cells'][n]['rankings'][method]['aurc'] for n in names
            ])) for method in output['cells'][names[0]]['rankings']},
        }
    return output
