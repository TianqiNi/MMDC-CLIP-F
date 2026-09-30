"""Separate artifact/inference review of completed screening; no optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors.torch import load_file

from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.cli import _manifest_binding
from mmdc_clip_f.research.view_risk.head import count_trainable_parameters
from mmdc_clip_f.research.view_risk.roles import Role, Operation, run_with_role_access
from mmdc_clip_f.research.view_risk.screening import (
    SCREENING_METHODS, choose_screening_epoch, matched_screening_model,
    screening_batch_from_tensors, slice_screening_batch, summarize_screening_scores,
    validate_screening_labels,
)
from mmdc_clip_f.research.view_risk.training import (
    build_training_schedule, load_role_manifest_for_operation, module_state_sha256,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('summary', 'private', 'plan', 'role-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(1)
    private, root = Path(args.private), Path(args.role_root)
    summary, plan = json.loads(Path(args.summary).read_text()), json.loads(Path(args.plan).read_text())
    execution = json.loads((private / 'execution.json').read_text())
    assert execution['status'] == 'PASS' and execution['completed_runs'] == 9
    assert execution['summary_sha256'] == sha256_file(args.summary)
    assert summary['provenance']['pilot_or_test_outcomes_accessed'] is False
    assert summary['provenance']['plan_sha256'] == sha256_file(args.plan)
    run = json.loads((private / 'run.json').read_text())
    source = run['source_commit']
    paths = sorted(str(x) for x in Path('src/mmdc_clip_f/research/view_risk').glob('*.py'))
    paths.append('evidence/phase5/signed-screening/run_screening.py')
    code_digests = [hashlib.sha256(subprocess.check_output(['git', 'show', f'{source}:{p}'])).hexdigest()
                    for p in paths]
    assert digest(code_digests) == run['binding']['code_sha256']
    records = {}
    for name, role, operation in [('confidence-fit', Role.CONFIDENCE_FIT, Operation.CONFIDENCE_FITTING),
                                  ('tune', Role.TUNE, Operation.TUNE_SELECTION)]:
        manifest = load_role_manifest_for_operation(root / f'{name}-role-manifest.json',
            expected=_manifest_binding(root / f'{name}-role-manifest-binding.json'),
            private_root=root, operation=operation, role=role)
        records[name] = run_with_role_access(manifest, operation=operation, roles=(role,), loader=tuple)
    index = json.loads((private / 'cache-bindings.json').read_text())
    assert len(index) == 77
    tune_batches = []
    for name, metadata in index.items():
        assert sha256_file(private / f'prepared/{name}.safetensors') == metadata['tensors_sha256']
        assert json.loads((private / f'prepared/{name}.json').read_text()) == metadata
        binding = metadata['binding']
        assert all(binding[k] == v for k, v in run['binding'].items())
        role_name = 'confidence-fit' if name.startswith('fit-') else 'tune'
        cohort = records[role_name]
        expected_sha = plan['confidence_fit_manifest_sha256' if role_name == 'confidence-fit' else 'tune_manifest_sha256']
        assert binding['manifest_sha256'] == expected_sha
        assert binding['ordered_rows_sha256'] == digest([r.exam_key for r in cohort])
        if role_name == 'confidence-fit':
            draws = build_training_schedule(cohort, epoch=binding['epoch'], seed=binding['seed'])
            assert binding['draws_sha256'] == digest([x.draw.to_dict() for x in draws])
        batch = screening_batch_from_tensors(load_file(str(private / f'prepared/{name}.safetensors')), 'vit_b_32')
        validate_screening_labels(batch, torch.tensor([r.density for r in cohort], dtype=torch.long))
        if role_name == 'tune':
            tune_batches.append((name, batch))
    tune_batches = [batch for _, batch in sorted(tune_batches)]
    assert len(tune_batches) == 17 and tune_batches[0].raw.batch_size == 370
    error = np.stack([batch.targets.observed_error.numpy() for batch in tune_batches])
    msp = np.stack([batch.raw.current_probabilities.max(1).values.numpy() for batch in tune_batches])
    assert summarize_screening_scores(error, msp) == summary['msp']
    assert summary['msp']['cells'][0]['accuracy'] == 290 / 370  # existing epoch-38 tune result
    reviewed_checkpoints, reproduced = 0, 0
    parameters = {}
    for seed in plan['seeds']:
        for method in SCREENING_METHODS:
            path = private / f'{method}-seed-{seed}'
            result = summary['results'][f'{method}:{seed}']
            history = json.loads((path / 'history.json').read_text())
            training = json.loads((path / 'training-result.json').read_text())
            assert result['history'] == history and result['training'] == training
            assert [x['epoch'] for x in history] == list(range(1, 21))
            assert training['update_count'] == 3300 and training['exposed_record_count'] == 987
            assert training['actual_exposure_verified'] is True
            assert choose_screening_epoch(history) == result['selected']
            assert history[-1] == result['fixed_epoch_20']
            head = matched_screening_model(method, seed, 'vit_b_32')
            parameters[method] = count_trainable_parameters(head)
            assert parameters[method] == training['parameter_count']
            resume_meta = json.loads((path / 'resume.json').read_text())
            assert sha256_file(path / 'resume.pt') == resume_meta['checkpoint_sha256']
            resume = torch.load(path / 'resume.pt', map_location='cpu', weights_only=True)
            assert resume['completed_epoch'] == 20 and resume['update_count'] == 3300
            for value in resume['model_state'].values():
                assert not value.is_floating_point() or torch.isfinite(value).all()
            for metric in history:
                checkpoint = path / f"epoch-{metric['epoch']:02d}.safetensors"
                assert sha256_file(checkpoint) == metric['checkpoint_sha256']
                state = load_file(str(checkpoint))
                head.load_state_dict(state, strict=True)
                assert all(not x.is_floating_point() or torch.isfinite(x).all() for x in state.values())
                saved_scores = load_file(str(path / f"scores-epoch-{metric['epoch']:02d}.safetensors"))
                recomputed = summarize_screening_scores(error, saved_scores['confidence'].numpy())
                assert recomputed['cells'] == metric['cells']
                if metric['epoch'] == 20:
                    assert module_state_sha256(head) == training['model_state_sha256']
                reviewed_checkpoints += 1
            head.to(args.device).eval()
            for epoch in sorted({20, result['selected']['epoch']}):
                head.load_state_dict(load_file(str(path / f'epoch-{epoch:02d}.safetensors'), device=args.device), strict=True)
                saved = load_file(str(path / f'scores-epoch-{epoch:02d}.safetensors'))
                confidence = []
                with torch.no_grad():
                    for batch in tune_batches:
                        scores = []
                        for begin in range(0, 370, 256):
                            current = slice_screening_batch(batch, slice(begin, begin + 256), device=args.device)
                            output = head(current.raw)
                            assert torch.equal(output.classifier_prediction, current.raw.current_prediction)
                            scores.append(output.confidence.cpu())
                        confidence.append(torch.cat(scores))
                torch.testing.assert_close(torch.stack(confidence), saved['confidence'], atol=1e-6, rtol=1e-6)
                reproduced += 1
    assert parameters['candidate'] == parameters['no_effect_supervision']
    output = {'status': 'PASS', 'execution_source_commit': source, 'cache_bindings_verified': 77,
        'checkpoint_states_and_metrics_verified': reviewed_checkpoints,
        'selected_or_final_checkpoint_inferences_reproduced': reproduced,
        'completed_runs': 9, 'total_verified_optimizer_updates': 29700,
        'parameter_counts': parameters, 'role_scope': ['confidence_fit', 'tune'],
        'pilot_or_test_outcomes_accessed': False, 'historical_source_hash_verified': True,
        'clean_accuracy_matches_previous_classifier_tune_result': True,
        'elapsed_seconds': time.monotonic() - start,
        'limitation': 'Orchestrator artifact review; not an independent external review or full pilot eligibility.'}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
