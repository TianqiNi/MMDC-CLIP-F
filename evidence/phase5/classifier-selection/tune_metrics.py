"""Aggregate development metrics for the frozen selected classifier, tune only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from mmdc_clip_f.cli import build_parser
from mmdc_clip_f.research.view_risk.cli import _manifest_binding
from mmdc_clip_f.research.view_risk.production import (
    load_selected_classifier_artifact, load_tensor_checkpoint_state,
    reload_verified_public_classifier, role_image_batch_loader,
)
from mmdc_clip_f.research.view_risk.roles import Operation, Role, run_with_role_access
from mmdc_clip_f.research.view_risk.training import (
    load_research_run_config, load_role_manifest_for_operation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    argv = json.loads(Path(args.command_file).read_text())["argv"]
    a = build_parser().parse_args(argv[argv.index("view-risk-select-classifier"):])
    config = load_research_run_config(a.config)
    selected = load_selected_classifier_artifact(a.output, expected_config=config)
    manifest = load_role_manifest_for_operation(
        a.manifest, expected=_manifest_binding(a.manifest_binding), private_root=a.private_root,
        operation=Operation.TUNE_SELECTION, role=Role.TUNE,
    )
    payload = json.loads(Path(a.output).read_text())["artifact"]
    model, tokens = reload_verified_public_classifier(selected.initialization, device=a.device)
    model.load_state_dict(load_tensor_checkpoint_state(selected.selected_checkpoint_path), strict=True)
    model.eval()

    def evaluate(records):
        labels, scores = [], []
        seen = set()
        nll_sum = 0.0
        with torch.no_grad():
            for batch in role_image_batch_loader(
                records, image_root=a.image_root, image_size=selected.classifier.image_size,
                optimizer=config.classifier_schedule.optimizer, augmentation=None,
                epoch=payload["selected_epoch"], seed=config.classifier_schedule.seed,
                training=False, image_workers=a.image_workers,
                prefetch_batches=a.prefetch_batches, pin_memory=a.device.startswith("cuda"),
            ):
                assert not seen.intersection(batch.exam_keys)
                seen.update(batch.exam_keys)
                prediction = model({k: v.to(a.device, non_blocking=True)
                                    for k, v in batch.payload.views.items()}, tokens)
                y = batch.payload.labels.to(a.device, non_blocking=True)
                nll_sum += float(torch.nn.functional.cross_entropy(prediction, y, reduction="sum"))
                labels.append(y.cpu())
                scores.append(prediction.cpu())
        assert seen == {r.exam_key for r in records} and len(seen) == 370
        y = torch.cat(labels)
        logits = torch.cat(scores)
        assert torch.isfinite(logits).all()
        predicted = logits.argmax(1)
        probability = logits.softmax(1)
        matrix = confusion_matrix(y, predicted, labels=[0, 1, 2, 3])
        support = matrix.sum(1)
        recall = np.divide(matrix.diagonal(), support, out=np.zeros(4), where=support != 0)
        nll = nll_sum / len(y)
        selected_record = next(r for r in payload["tune_checkpoints"]
                               if r["epoch"] == payload["selected_epoch"])
        assert abs(nll - selected_record["tune_nll"]) < 1e-7
        one_hot = torch.nn.functional.one_hot(y, num_classes=4)
        return {
            "role": "tune", "development_only": True, "patients": len(y),
            "selected_epoch": payload["selected_epoch"],
            "selected_artifact_sha256": selected.sha256,
            "selected_checkpoint_sha256": selected.classifier.checkpoint_sha256,
            "selection_metric": "multiclass_nll", "nll": nll,
            "accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1, 2, 3],
                                       average="macro", zero_division=0)),
            "balanced_accuracy_present_classes": float(recall[support > 0].mean()),
            "quadratic_weighted_kappa": float(cohen_kappa_score(
                y, predicted, labels=[0, 1, 2, 3], weights="quadratic")),
            "multiclass_brier_sum_over_classes": float(((probability - one_hot) ** 2).sum(1).mean()),
            "class_order": ["A", "B", "C", "D"], "class_support": support.tolist(),
            "per_class_recall": recall.tolist(),
            "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "tune_majority_class_prevalence": float(support.max() / support.sum()),
            "checkpoint_nll": [{"epoch": r["epoch"], "nll": r["tune_nll"]}
                               for r in payload["tune_checkpoints"]],
            "interpretation": "Selection-set diagnostics; not held-out generalization or uncertainty-method evidence.",
        }

    result = run_with_role_access(manifest, operation=Operation.TUNE_SELECTION,
                                  roles=(Role.TUNE,), loader=evaluate)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoint_nll"}), flush=True)


if __name__ == "__main__":
    main()
