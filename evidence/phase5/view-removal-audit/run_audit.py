"""Run the predeclared confidence-fit support audit; keep row-level data private."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from safetensors.torch import save_file

from mmdc_clip_f.data import build_transforms
from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.cache import _require_external_or_ignored_destination
from mmdc_clip_f.research.view_risk.cli import _manifest_binding
from mmdc_clip_f.research.view_risk.image_pipeline import _record_fields, image_worker_pool
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.production import (
    default_private_image_reader, load_selected_classifier_artifact,
    load_tensor_checkpoint_state, reload_verified_public_classifier,
)
from mmdc_clip_f.research.view_risk.roles import Operation, Role, run_with_role_access
from mmdc_clip_f.research.view_risk.support_audit import (
    audit_cells, evaluate_removals, prepare_record, summarize_cells,
)
from mmdc_clip_f.research.view_risk.training import (
    load_research_run_config, load_role_manifest_for_operation, state_dict_sha256,
)


def main():
    parser = argparse.ArgumentParser()
    for flag in ['config', 'selected', 'manifest', 'manifest-binding', 'private-root',
                 'image-root', 'private-output', 'summary-output', 'plan']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(1)
    start = time.monotonic()
    plan = json.loads(Path(args.plan).read_text())
    config = load_research_run_config(args.config)
    manifest = load_role_manifest_for_operation(
        args.manifest, expected=_manifest_binding(args.manifest_binding),
        private_root=args.private_root, operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
    )
    selected = load_selected_classifier_artifact(args.selected, expected_config=config)
    assert selected.sha256 == plan['selected_artifact_sha256']
    assert manifest.manifest_sha256 in selected.readiness.manifest_sha256s
    assert selected.initialization.dataset == 'RSNA'
    private = Path(args.private_output).resolve()
    _require_external_or_ignored_destination(private / 'view-logits.safetensors')
    private.mkdir(parents=True, exist_ok=False)
    model, tokens = reload_verified_public_classifier(selected.initialization, device=args.device)
    model.load_state_dict(load_tensor_checkpoint_state(selected.selected_checkpoint_path), strict=True)
    model.eval().requires_grad_(False)
    initial_state = state_dict_sha256(model.state_dict())
    print('Selected classifier verified; confidence-fit extraction starting.', flush=True)

    def authorized(records):
        assert len(records) == plan['expected_exams']
        assert len({r.patient_key for r in records}) == plan['expected_patients']
        assert len({r.exam_key for r in records}) == len(records)
        assert all(r.role == Role.CONFIDENCE_FIT for r in records)
        batch_size = config.classifier_schedule.optimizer.batch_size
        all_logits, clean_legacy_scores = [], []
        clean_pixels_equal = False
        forward_equal = []
        with torch.no_grad(), image_worker_pool(args.workers) as pool:
            if pool is None:
                raise ValueError('this runner requires positive worker count')
            text = model._text_features(tokens)
            scale = model.clip_model.logit_scale.exp()
            pending = deque()
            next_index = 0
            completed = 0
            while pending or next_index < len(records):
                while len(pending) < 4 and next_index < len(records):
                    group = records[next_index:next_index + batch_size]
                    pending.append((group, [pool.submit(
                        prepare_record, _record_fields(r), args.image_root,
                        selected.classifier.image_size, plan['seed'],
                    ) for r in group]))
                    next_index += len(group)
                group, futures = pending.popleft()
                images = torch.stack([future.result() for future in futures])
                if completed == 0:
                    _, transform = build_transforms(selected.classifier.image_size, None)
                    reference = torch.stack([transform(default_private_image_reader(
                        group[0], v, args.image_root)) for v in CANONICAL_VIEWS])
                    assert torch.equal(images[0, 0], reference)
                    clean_pixels_equal = True
                variant_logits = []
                for variant in range(5):
                    views = {v: images[:, variant, i].contiguous().pin_memory().to(
                        args.device, non_blocking=True) for i, v in enumerate(CANONICAL_VIEWS)}
                    encoded = model.clip_model.get_image_features(
                        pixel_values=torch.cat([views[v] for v in model.input_order], dim=0))
                    chunks = dict(zip(model.input_order, encoded.split(len(group))))
                    logits = {v: scale * (x / x.norm(p=2, dim=-1, keepdim=True)) @ text.t()
                              for v, x in chunks.items()}
                    legacy = model.fuse_logits(logits)
                    if variant == 0:
                        clean_legacy_scores.append(legacy.cpu())
                    if completed == 0:
                        direct = model(views, tokens)
                        assert torch.equal(legacy, direct)
                        forward_equal.append(True)
                    variant_logits.append(torch.stack([logits[v].cpu()
                                                       for v in CANONICAL_VIEWS], dim=1))
                all_logits.append(torch.stack(variant_logits, dim=1))
                completed += len(group)
                if completed % 99 == 0 or completed == len(records):
                    print(json.dumps({'completed_exams': completed, 'total': len(records),
                                      'elapsed_seconds': time.monotonic() - start}), flush=True)
        logits = torch.cat(all_logits)
        labels = torch.tensor([r.density for r in records], dtype=torch.long)
        assert logits.shape == (len(records), 5, 4, 4) and torch.isfinite(logits).all()
        save_file({'view_logits': logits, 'labels': labels}, str(private / 'view-logits.safetensors'))
        rows = {'exam_keys': [r.exam_key for r in records],
                'patient_keys': [r.patient_key for r in records],
                'manifest_sha256': manifest.manifest_sha256,
                'selected_artifact_sha256': selected.sha256,
                'plan_sha256': sha256_file(args.plan), 'view_order': list(CANONICAL_VIEWS)}
        (private / 'rows.json').write_text(json.dumps(rows) + '\n')
        cells = {}
        for name, variant, stressed in audit_cells():
            per_view = {v: logits[:, variant if v in stressed else 0, i].to(args.device)
                        for i, v in enumerate(CANONICAL_VIEWS)}
            cells[name] = evaluate_removals(per_view, labels.to(args.device))
        # Fusion is per patient, so vectorizing the audit must preserve the
        # original minibatch classifier outputs exactly on this real cohort.
        assert torch.equal(cells['clean']['parent_scores'], torch.cat(clean_legacy_scores))
        save_file({f'{index}.{key}': value.contiguous()
                   for index, cell in enumerate(cells.values()) for key, value in cell.items()},
                  str(private / 'removal-results.safetensors'))
        result = summarize_cells(cells, labels, rows['patient_keys'])
        assert state_dict_sha256(model.state_dict()) == initial_state
        assert sha256_file(selected.selected_checkpoint_path) == selected.classifier.checkpoint_sha256
        result['provenance'] = {
            'role': 'confidence_fit', 'selected_epoch': 38,
            'selected_artifact_sha256': selected.sha256,
            'checkpoint_sha256': selected.classifier.checkpoint_sha256,
            'manifest_sha256': manifest.manifest_sha256, 'config_sha256': config.sha256,
            'plan_sha256': sha256_file(args.plan),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'frozen_model_unchanged': True, 'clean_preprocessing_exact': clean_pixels_equal,
            'first_batch_original_forward_exact_all_five_variants': all(forward_equal),
            'all_clean_fused_scores_equal_original_minibatches': True,
            'private_logits_sha256': sha256_file(private / 'view-logits.safetensors'),
            'private_rows_sha256': sha256_file(private / 'rows.json'),
            'private_removal_results_sha256': sha256_file(private / 'removal-results.safetensors'),
            'pilot_or_test_outcomes_accessed': False, 'elapsed_seconds': time.monotonic() - start,
        }
        return result

    result = run_with_role_access(manifest, operation=Operation.CONFIDENCE_FITTING,
                                  roles=(Role.CONFIDENCE_FIT,), loader=authorized)
    Path(args.summary_output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'patients': result['patients'],
                      'strata': result['strata']}), flush=True)


if __name__ == '__main__':
    main()
