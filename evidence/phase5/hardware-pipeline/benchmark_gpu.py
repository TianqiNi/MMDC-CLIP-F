"""Compare actual resumed classifier updates; never write training checkpoints."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from mmdc_clip_f.cli import build_parser
from mmdc_clip_f.research.view_risk.cli import _manifest_binding
from mmdc_clip_f.research.view_risk.production import (
    load_public_initialization_artifact, reload_verified_public_classifier,
    role_image_batch_loader,
)
from mmdc_clip_f.research.view_risk.roles import Operation, Role
from mmdc_clip_f.research.view_risk.training import (
    TrainingBinding, _load_resume, load_research_run_config,
    load_role_manifest_for_operation, state_dict_sha256,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    argv = json.loads(Path(args.command_file).read_text())["argv"]
    a = build_parser().parse_args(argv[argv.index("view-risk-fit-classifier"):])
    config = load_research_run_config(a.config)
    schedule = config.classifier_schedule
    manifest = load_role_manifest_for_operation(
        a.manifest, expected=_manifest_binding(a.manifest_binding), private_root=a.private_root,
        operation=Operation.CLASSIFIER_FITTING, role=Role.CLASSIFIER_FIT,
    )
    initialization = load_public_initialization_artifact(a.initialization)
    binding = TrainingBinding(**json.loads(
        Path(args.resume_checkpoint).with_suffix(".json").read_text())["binding"])
    results = []
    for workers in (0, 12):
        model, tokens = reload_verified_public_classifier(initialization, device="cuda:0")
        optimizer = torch.optim.Adam(model.parameters(), lr=schedule.optimizer.learning_rate,
                                     weight_decay=schedule.optimizer.weight_decay)
        epoch, updates = _load_resume(args.resume_checkpoint, model=model,
                                     optimizer=optimizer, binding=binding)
        assert epoch == 15 and updates == 16050
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        iterator = iter(role_image_batch_loader(
            manifest.records, image_root=a.image_root, image_size=224,
            optimizer=schedule.optimizer, augmentation=schedule.augmentation,
            epoch=epoch + 1, seed=schedule.seed, training=True,
            image_workers=workers, prefetch_batches=2, pin_memory=workers > 0,
        ))
        waits, events = [], []
        for _ in range(args.steps):
            ready = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - ready)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            views = {k: v.to("cuda:0", non_blocking=workers > 0)
                     for k, v in batch.payload.views.items()}
            labels = batch.payload.labels.to("cuda:0", non_blocking=workers > 0)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(views, tokens), labels)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            end.record()
            events.append((begin, end))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        iterator.close()
        result = {"workers": workers, "steps": args.steps, "seconds": elapsed,
                  "first_batch_seconds": waits[0],
                  "mean_later_data_wait_seconds": sum(waits[1:]) / (len(waits) - 1),
                  "gpu_step_seconds": sum(b.elapsed_time(e) for b, e in events) / 1000,
                  "model_sha256": state_dict_sha256(model.state_dict()),
                  "optimizer_sha256": state_dict_sha256({
                      f"{k}/{name}": v for k, state in optimizer.state_dict()["state"].items()
                      for name, v in state.items() if isinstance(v, torch.Tensor)}),
                  "rng_sha256": state_dict_sha256({"cpu": torch.get_rng_state(),
                                                   "cuda": torch.cuda.get_rng_state()})}
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, loss, views, labels, batch, tokens, iterator
        gc.collect()
        torch.cuda.empty_cache()
    equal = all(results[0][k] == results[1][k]
                for k in ("model_sha256", "optimizer_sha256", "rng_sha256"))
    output = {"starting_epoch": 15, "role": "classifier_fit", "runs": results,
              "exact_model_optimizer_rng_match": equal,
              "speedup_including_worker_startup": results[0]["seconds"] / results[1]["seconds"]}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    if not equal:
        raise RuntimeError("actual resumed GPU updates differ")


if __name__ == "__main__":
    main()
