"""Run the prespecified development screen; all patient-level artifacts stay private."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file
from torchvision import transforms

from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.cache import _require_external_or_ignored_destination
from mmdc_clip_f.research.view_risk.cli import _manifest_binding
from mmdc_clip_f.research.view_risk.features import (
    LEGACY_IMAGE_MEAN, LEGACY_IMAGE_STD, load_verified_frozen_encoder, tensor_sha256,
)
from mmdc_clip_f.research.view_risk.head import compute_view_risk_loss, count_trainable_parameters
from mmdc_clip_f.research.view_risk.head_inputs import prepare_raw_head_inputs
from mmdc_clip_f.research.view_risk.image_pipeline import _record_fields, image_worker_pool
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.perturbations import realize_parent, realize_training_parent
from mmdc_clip_f.research.view_risk.production import (
    _atomic_safetensors, default_private_image_reader,
    load_selected_classifier_artifact, reload_verified_public_classifier, save_tensor_checkpoint,
    load_tensor_checkpoint_state,
)
from mmdc_clip_f.research.view_risk.roles import Operation, PrivateExamRecord, Role, run_with_role_access
from mmdc_clip_f.research.view_risk.screening import (
    SCREENING_METHODS, choose_screening_epoch, matched_screening_model,
    pack_screening_batch, screening_batch_from_tensors, screening_tensors,
    slice_screening_batch, summarize_screening_scores, validate_screening_labels,
)
from mmdc_clip_f.research.view_risk.targets import build_intervention_targets
from mmdc_clip_f.research.view_risk.training import (
    RoleBoundBatch, TrainingBinding, TrainingResult, build_training_schedule, expected_tune_cache_realization,
    fit_role_bound_module, load_research_run_config, load_role_manifest_for_operation,
    tune_selection_cells, _load_resume, module_state_sha256,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value, *, public_aggregate=False):
    """Replace task-owned progress atomically; patient-bearing paths stay private."""
    staging = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    if not public_aggregate:
        for candidate in (path, staging):
            _require_external_or_ignored_destination(candidate)
    try:
        with staging.open('x') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        staging.chmod(0o644 if public_aggregate else 0o600)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def decode_record(fields, root, size):
    record = PrivateExamRecord(**fields)
    if record.role not in (Role.CONFIDENCE_FIT, Role.TUNE):
        raise PermissionError('screening image role is unauthorized')
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return torch.stack([transform(default_private_image_reader(record, v, root)) for v in CANONICAL_VIEWS])


def authorized_records(manifest, operation, role, expected_count):
    def access(records):
        if len(records) != expected_count or len({r.patient_key for r in records}) != expected_count:
            raise ValueError('screening cohort differs from the prespecified exam/patient count')
        if any(r.role != role for r in records) or len({r.exam_key for r in records}) != len(records):
            raise PermissionError('screening cohort contains duplicate or unauthorized rows')
        return tuple(records)
    return run_with_role_access(manifest, operation=operation, roles=(role,), loader=access)


def extract_batch(encoder, records, pixels, draws, specs, device):
    """Actual realized pixels and exact targets, reordered to the authorized manifest."""
    grouped = defaultdict(list)
    parents = []
    for i, record in enumerate(records):
        images = dict(zip(CANONICAL_VIEWS, pixels[i]))
        if draws is not None:
            if draws[i].exam_key != record.exam_key:
                raise ValueError('training schedule row order is stale')
            parent = realize_training_parent(images, draws[i].draw)
        else:
            observed, per_view = specs[i]
            parent = realize_parent(images, observed, perturbations=per_view)
        parents.append(parent)
        grouped[parent.observed_views].append(i)
    normalized = []
    input_digest = hashlib.sha256()
    for observed, indices in grouped.items():
        mapping = {}
        for view in observed:
            images = torch.stack([parents[i].images[view] for i in indices])
            mean = images.new_tensor(LEGACY_IMAGE_MEAN)[None, :, None, None]
            std = images.new_tensor(LEGACY_IMAGE_STD)[None, :, None, None]
            value = ((images - mean) / std).contiguous()
            input_digest.update(tensor_sha256(value).encode())
            mapping[view] = (value.pin_memory().to(device, non_blocking=True)
                             if str(device).startswith('cuda') else value.to(device))
        normalized.append(mapping)
    features = encoder.extract_normalized_batches(normalized, max_images=12)
    labels = torch.tensor([r.density for r in records], device=device, dtype=torch.long)
    groups = []
    for feature, indices in zip(features, grouped.values()):
        target = build_intervention_targets(feature.logits_by_view, labels[indices],
                                            feature.observed_views, fusion_pairs=feature.fusion_pairs)
        groups.append((prepare_raw_head_inputs(feature, backbone=encoder.identity.backbone), target))
    batch = pack_screening_batch(groups, list(grouped.values()))
    validate_screening_labels(batch, labels)
    return screening_batch_from_tensors(screening_tensors(batch), encoder.identity.backbone), input_digest.hexdigest()


def cached_batch(private, name, binding, producer, labels, backbone):
    tensor_path, meta_path = private / f'{name}.safetensors', private / f'{name}.json'
    if tensor_path.exists() or meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata['binding'] != binding or sha256_file(tensor_path) != metadata['tensors_sha256']:
            raise ValueError('screening feature cache is stale or modified')
        batch = screening_batch_from_tensors(load_file(str(tensor_path)), backbone)
    else:
        batch, input_digest = producer()
        _atomic_safetensors(tensor_path, screening_tensors(batch))
        metadata = {'binding': binding, 'tensors_sha256': sha256_file(tensor_path),
                    'realized_normalized_input_sha256': input_digest}
        write_json(meta_path, metadata)
    validate_screening_labels(batch, labels)
    return batch, metadata


def score(model, cells, device):
    confidence, raw_effect, reported_effect = [], [], []
    model.eval()
    with torch.no_grad():
        for cell in cells:
            cs, rs, ps = [], [], []
            for begin in range(0, cell.raw.batch_size, 256):
                batch = slice_screening_batch(cell, slice(begin, begin + 256), device=device)
                output = model(batch.raw)  # no labels, effects or perturbation descriptors
                if not torch.equal(output.classifier_prediction, batch.raw.current_prediction):
                    raise RuntimeError('confidence head changed classifier predictions')
                cs.append(output.confidence.cpu())
                rs.append(output.raw_auxiliary_logits.argmax(-1).cpu())
                ps.append(output.reported_auxiliary_probabilities.argmax(-1).cpu())
            confidence.append(torch.cat(cs))
            raw_effect.append(torch.cat(rs))
            reported_effect.append(torch.cat(ps))
    return {'confidence': torch.stack(confidence), 'raw_effect': torch.stack(raw_effect),
            'reported_effect': torch.stack(reported_effect)}


def main():
    parser = argparse.ArgumentParser()
    for flag in ('config', 'plan', 'selected', 'role-root', 'image-root', 'private-output', 'summary-output'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=int, default=18000)
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(1)
    start = time.monotonic()
    private = Path(args.private_output).resolve()
    _require_external_or_ignored_destination(private / 'run.json')
    private.mkdir(parents=True, exist_ok=True)
    lock = (private / 'process.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = json.loads(Path(args.plan).read_text())
        config = load_research_run_config(args.config)
        if (config.sha256 != plan['config_sha256'] or tuple(plan['methods']) != SCREENING_METHODS
            or tuple(config.seeds) != tuple(plan['seeds']) or config.epochs != plan['epochs']
            or asdict(config.optimizer) != plan['optimizer']):
            raise ValueError('screening plan disagrees with the unchanged study settings')
        root = Path(args.role_root).resolve()
        manifests = {}
        cohorts = {}
        for label, role, operation in [('confidence-fit', Role.CONFIDENCE_FIT, Operation.CONFIDENCE_FITTING),
                                       ('tune', Role.TUNE, Operation.TUNE_SELECTION)]:
            manifests[label] = load_role_manifest_for_operation(
                root / f'{label}-role-manifest.json',
                expected=_manifest_binding(root / f'{label}-role-manifest-binding.json'),
                private_root=root, operation=operation, role=role)
            key = 'confidence_fit' if label == 'confidence-fit' else 'tune'
            if manifests[label].manifest_sha256 != plan[f'{key}_manifest_sha256']:
                raise ValueError('screening manifest differs from the frozen plan')
            cohorts[label] = authorized_records(manifests[label], operation, role,
                plan['expected_fit_exams' if label == 'confidence-fit' else 'expected_tune_exams'])
        fit, tune = cohorts['confidence-fit'], cohorts['tune']
        if {r.patient_key for r in fit} & {r.patient_key for r in tune}:
            raise PermissionError('fit/tune patient roles overlap')
        selected = load_selected_classifier_artifact(args.selected, expected_config=config)
        if selected.sha256 != plan['selected_artifact_sha256'] or (
            selected.classifier.checkpoint_sha256 != plan['classifier_checkpoint_sha256']):
            raise ValueError('selected classifier differs from the frozen screening plan')
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        code_files = sorted(Path('src/mmdc_clip_f/research/view_risk').glob('*.py')) + [Path(__file__)]
        code_sha = digest([sha256_file(p) for p in code_files])
        run_binding = {'version': plan['version'], 'plan_sha256': sha256_file(args.plan),
                       'config_sha256': config.sha256, 'selected_artifact_sha256': selected.sha256,
                       'encoder_checkpoint_sha256': selected.classifier.checkpoint_sha256,
                       'code_sha256': code_sha}
        if (private / 'run.json').exists():
            if json.loads((private / 'run.json').read_text())['binding'] != run_binding:
                raise ValueError('cannot resume a screening run after code/plan/binding changes')
        else:
            write_json(private / 'run.json', {'binding': run_binding, 'source_commit': source})
        print('Role and selected-classifier provenance verified. Preparing private caches.', flush=True)
        model, tokens = reload_verified_public_classifier(selected.initialization, device=args.device)
        encoder = load_verified_frozen_encoder(model, tokens, selected.selected_checkpoint_path,
                                              backbone=config.backbone, prompts=selected.classifier.prompts)
        del model, tokens
        pixels = {}
        with image_worker_pool(args.workers) as pool:
            if pool is None:
                raise ValueError('screening requires positive image worker count')
            for label, records in cohorts.items():
                futures = [pool.submit(decode_record, _record_fields(r), args.image_root,
                                       selected.classifier.image_size) for r in records]
                pixels[label] = torch.stack([f.result() for f in futures])
        tune_labels = torch.tensor([r.density for r in tune], dtype=torch.long)
        fit_labels = torch.tensor([r.density for r in fit], dtype=torch.long)
        cache = private / 'prepared'
        cache.mkdir(exist_ok=True)
        tune_batches, cache_bindings = [], {}
        # One direct real-image forward checks batched extraction against the
        # original classifier before any optimization or confidence selection.
        normalized = {v: transforms.Normalize(LEGACY_IMAGE_MEAN, LEGACY_IMAGE_STD)(
            pixels['tune'][:3, i]).to(args.device) for i, v in enumerate(CANONICAL_VIEWS)}
        reference = encoder.extract_normalized(normalized)
        bulk = encoder.extract_normalized_batches([normalized])[0]
        with torch.no_grad():
            direct = encoder.classifier(normalized, encoder.input_ids)
        for actual, expected in zip(bulk.all_tensors(), reference.all_tensors()):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(bulk.scores, direct, atol=1e-6, rtol=1e-6)
        if not torch.equal(bulk.prediction, direct.argmax(1)):
            raise RuntimeError('bulk extraction changed real-image classifier decisions')
        del normalized, reference, bulk, direct
        for index, cell in enumerate(tune_selection_cells()):
            specs = [expected_tune_cache_realization(r, cell, stress_seed=config.stress_seed) for r in tune]
            binding = {**run_binding, 'role': 'tune', 'manifest_sha256': manifests['tune'].manifest_sha256,
                       'cell': list(cell), 'ordered_rows_sha256': digest([r.exam_key for r in tune]),
                       'realizations_sha256': digest([{v: s.to_dict() for v, s in per.items()} for _, per in specs])}
            batch, metadata = cached_batch(cache, f'tune-{index:02d}', binding,
                lambda: extract_batch(encoder, tune, pixels['tune'], None, specs, args.device),
                tune_labels, config.backbone)
            tune_batches.append(batch)
            cache_bindings[f'tune-{index:02d}'] = metadata
            print(json.dumps({'prepared_tune_cells': index + 1, 'total': 17,
                              'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
        errors = torch.stack([x.targets.observed_error for x in tune_batches]).numpy()
        msp = torch.stack([x.raw.current_probabilities.max(1).values for x in tune_batches]).numpy()
        msp_metrics = summarize_screening_scores(errors, msp)
        results = {}
        for seed in plan['seeds']:
            for method in SCREENING_METHODS:
                run = private / f'{method}-seed-{seed}'
                run.mkdir(exist_ok=True)
                result_path = run / 'training-result.json'
                history_path = run / 'history.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    history = json.loads(history_path.read_text())
                    if result['completed_epoch'] != plan['epochs'] or len(history) != plan['epochs']:
                        raise ValueError('completed screening run has an incomplete epoch budget')
                else:
                    head = matched_screening_model(method, seed, config.backbone).to(args.device)
                    history = json.loads(history_path.read_text()) if history_path.exists() else []
                    binding = TrainingBinding(config.protocol_sha256, config.sha256,
                        config.search_table.sha256, manifests['confidence-fit'].manifest_sha256,
                        selected.classifier.checkpoint_sha256, method, seed)

                    def loader(records, epoch):
                        draws = build_training_schedule(records, epoch=epoch, seed=seed)
                        name = f'fit-seed-{seed}-epoch-{epoch:02d}'
                        cache_binding = {**run_binding, 'role': 'confidence_fit',
                            'manifest_sha256': manifests['confidence-fit'].manifest_sha256,
                            'seed': seed, 'epoch': epoch,
                            'ordered_rows_sha256': digest([r.exam_key for r in records]),
                            'draws_sha256': digest([x.draw.to_dict() for x in draws])}
                        batch, metadata = cached_batch(cache, name, cache_binding,
                            lambda: extract_batch(encoder, records, pixels['confidence-fit'], draws, None, args.device),
                            fit_labels, config.backbone)
                        cache_bindings[name] = metadata
                        for begin in range(0, len(records), config.optimizer.batch_size):
                            stop = begin + config.optimizer.batch_size
                            yield RoleBoundBatch(tuple(r.exam_key for r in records[begin:stop]),
                                slice_screening_batch(batch, slice(begin, stop), device=args.device))

                    def loss_fn(module, batch):
                        return compute_view_risk_loss(module(batch.raw), batch.targets,
                            auxiliary_weight=plan['auxiliary_weights'][method]).total

                    def callback(epoch, module):
                        checkpoint = run / f'epoch-{epoch:02d}.safetensors'
                        if checkpoint.exists():
                            saved = load_tensor_checkpoint_state(checkpoint)
                            if set(saved) != set(module.state_dict()) or any(
                                not torch.equal(saved[k], x.detach().cpu()) for k, x in module.state_dict().items()
                            ):
                                raise ValueError('existing epoch checkpoint differs from verified resume state')
                        else:
                            save_tensor_checkpoint(module, checkpoint)
                        def evaluate(records):
                            if tuple(r.exam_key for r in records) != tuple(r.exam_key for r in tune):
                                raise ValueError('tune row identities changed')
                            return score(module, tune_batches, args.device)
                        predictions = run_with_role_access(manifests['tune'], operation=Operation.TUNE_SELECTION,
                                                          roles=(Role.TUNE,), loader=evaluate)
                        scores_path = run / f'scores-epoch-{epoch:02d}.safetensors'
                        if scores_path.exists():
                            saved_scores = load_file(str(scores_path))
                            if set(saved_scores) != set(predictions):
                                raise ValueError('existing epoch score schema is invalid')
                            for name, value in predictions.items():
                                torch.testing.assert_close(saved_scores[name], value, atol=1e-6, rtol=1e-6)
                        else:
                            _atomic_safetensors(scores_path, predictions)
                        metric = summarize_screening_scores(errors, predictions['confidence'].numpy())
                        metric.update({'epoch': epoch, 'checkpoint_sha256': sha256_file(checkpoint)})
                        history[:] = [x for x in history if x['epoch'] < epoch] + [metric]
                        write_json(history_path, history)
                        module.train()
                        print(json.dumps({'method': method, 'seed': seed, 'completed_epoch': epoch,
                            'clean_aurc': metric['clean_aurc'], 'balanced_aurc': metric['balanced_aurc'],
                            'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
                        if time.monotonic() - start > args.max_seconds and epoch < plan['epochs']:
                            raise TimeoutError('screening time budget reached after a saved epoch')

                    resumed = (run / 'resume.json').exists()
                    completed, updates = 0, 0
                    if resumed:
                        optimizer = torch.optim.Adam(head.parameters(), lr=config.optimizer.learning_rate,
                                                     weight_decay=config.optimizer.weight_decay)
                        completed, updates = _load_resume(run / 'resume.pt', model=head,
                                                         optimizer=optimizer, binding=binding)
                        del optimizer
                        if [x['epoch'] for x in history] == list(range(1, completed)):
                            callback(completed, head)  # repair an interrupted evaluation; no optimizer step
                        elif [x['epoch'] for x in history] != list(range(1, completed + 1)):
                            raise ValueError('resume history is incomplete beyond the last callback boundary')
                    if completed == plan['epochs']:
                        result = asdict(TrainingResult(method=method, seed=seed, completed_epoch=completed,
                            update_count=updates, exposed_record_count=len(fit),
                            parameter_count=count_trainable_parameters(head), model_state_sha256=module_state_sha256(head),
                            manifest_sha256=manifests['confidence-fit'].manifest_sha256,
                            scaling='declared_identity_no_fitted_scaler', selection_trial_budget=plan['epochs'],
                            actual_exposure_verified=True))
                    else:
                        result = asdict(fit_role_bound_module(
                            manifest=manifests['confidence-fit'], operation=Operation.CONFIDENCE_FITTING,
                            role=Role.CONFIDENCE_FIT, model=head, optimizer_config=config.optimizer,
                            epochs=plan['epochs'], binding=binding, batch_loader=loader, loss_fn=loss_fn,
                            checkpoint_path=run / 'resume.pt', resume=resumed, epoch_callback=callback))
                    write_json(result_path, result)
                    del head
                if [x['epoch'] for x in history] != list(range(1, plan['epochs'] + 1)) or result['update_count'] != (
                    (len(fit) + config.optimizer.batch_size - 1) // config.optimizer.batch_size * plan['epochs']
                ):
                    raise ValueError('screening epoch/checkpoint or optimizer exposure budget is incomplete')
                chosen = choose_screening_epoch(history)
                results[f'{method}:{seed}'] = {'training': result, 'selected': chosen,
                    'fixed_epoch_20': history[-1], 'history': history}
                if time.monotonic() - start > args.max_seconds:
                    raise TimeoutError('screening time budget reached after a completed method')
        encoder._assert_text_inputs_unchanged()
        encoder._assert_weights_unchanged(verify_content=True)
        write_json(private / 'cache-bindings.json', cache_bindings)
        summary = {'version': plan['version'], 'evidence_kind': 'development_screening_only',
            'provenance': {**run_binding, 'source_commit': source,
                'fit_exams': len(fit), 'tune_exams': len(tune), 'tune_cells': 17,
                'frozen_state_verified_unchanged': True, 'real_image_bulk_vs_original_forward_verified': True,
                'pilot_or_test_outcomes_accessed': False,
                'elapsed_seconds': time.monotonic() - start,
                'private_cache_bindings_sha256': sha256_file(private / 'cache-bindings.json')},
            'msp': msp_metrics, 'results': results}
        write_json(Path(args.summary_output), summary, public_aggregate=True)
        write_json(private / 'execution.json', {'status': 'PASS', 'completed_runs': len(results),
                     'elapsed_seconds': time.monotonic() - start,
                     'summary_sha256': sha256_file(args.summary_output)})
        print(json.dumps({'status': 'PASS', 'completed_runs': len(results)}), flush=True)
    except BaseException as exc:
        (private / 'error.log').write_text(traceback.format_exc())
        write_json(private / 'execution.json', {'status': 'STOPPED' if isinstance(exc, TimeoutError) else 'FAIL',
                     'error_type': type(exc).__name__, 'elapsed_seconds': time.monotonic() - start})
        print(json.dumps({'status': 'STOPPED' if isinstance(exc, TimeoutError) else 'FAIL',
                          'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    finally:
        lock.close()


if __name__ == '__main__':
    main()
