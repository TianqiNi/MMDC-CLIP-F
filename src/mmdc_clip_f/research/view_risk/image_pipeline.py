"""Bounded CPU image workers with the serial loader's exact RNG sequence.

Speculative batches use private copies of the parent's CPU RNG. Their state is
committed only when delivered. If a consumer draws CPU random numbers between
batches, speculative work is discarded and regenerated from the new state.
CUDA RNG is never touched. Workers replay torchvision itself on full-size images.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context

import torch
from PIL import Image
from torchvision import transforms

from mmdc_clip_f.data import build_transforms

from .inputs import CANONICAL_VIEWS
from .roles import PrivateExamRecord


def _worker_init():
    # Avoid multiplying CPU thread pools by the number of decoding workers.
    torch.set_num_threads(1)


@contextmanager
def image_worker_pool(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("image_workers must be a nonnegative integer")
    if workers == 0:
        yield None
        return
    # Never fork a process containing an initialized CUDA context.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=_worker_init,
    ) as pool:
        yield pool


def _prepare_image(fields, view, root, image_size, augmentation, before, after):
    from .production import default_private_image_reader

    record = PrivateExamRecord(**fields)
    transform, _ = build_transforms(image_size, augmentation)
    image = default_private_image_reader(record, view, root)
    torch.set_rng_state(torch.from_numpy(before.copy()))
    tensor = transform(image)
    if not torch.equal(torch.get_rng_state(), torch.from_numpy(after)):
        raise RuntimeError("parallel preprocessing RNG consumption differs from serial")
    return tensor.numpy()


def _record_fields(record):
    return {
        "dataset_namespace": record.dataset_namespace, "exam_key": record.exam_key,
        "patient_key": record.patient_key, "density": record.density,
        "source_manifest": record.source_manifest, "views": dict(record.views),
        "role": record.role,
    }


def parallel_image_batches(
    batches, *, pool, image_root, image_size, augmentation, prefetch_batches,
):
    if (isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int)
            or not 1 <= prefetch_batches <= 8):
        raise ValueError("prefetch_batches must be an integer between 1 and 8")
    transform, _ = build_transforms(image_size, augmentation)
    # In supported torchvision 0.18, RandAugment's random draws depend only on
    # operation choice/sign, never pixels or dimensions. Run it on a tiny RGB
    # placeholder to reserve its draws; workers verify the resulting RNG state.
    random_steps = []
    for step in transform.transforms:
        if type(step) is transforms.RandAugment:
            random_steps.append(step)
        elif type(step) not in (transforms.Resize, transforms.ToTensor, transforms.Normalize):
            raise ValueError("parallel preprocessing encountered an unsupported transform")
    placeholder = Image.new("RGB", (8, 8))
    pending = deque()
    next_index = 0

    def submit(index, initial_state):
        records = batches[index]
        futures = []
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial_state)
            for view in CANONICAL_VIEWS:
                for record in records:
                    before = torch.get_rng_state().numpy().copy()
                    for step in random_steps:
                        step(placeholder)
                    after = torch.get_rng_state().numpy().copy()
                    futures.append(pool.submit(
                        _prepare_image, _record_fields(record), view, str(image_root),
                        image_size, augmentation, before, after,
                    ))
            final_state = torch.get_rng_state().clone()
        return index, initial_state, final_state, futures

    try:
        while pending or next_index < len(batches):
            current_state = torch.get_rng_state()
            if pending and not torch.equal(pending[0][1], current_state):
                next_index = pending[0][0]
                for _, _, _, futures in pending:
                    for future in futures:
                        future.cancel()
                pending.clear()
            while len(pending) < prefetch_batches and next_index < len(batches):
                initial = pending[-1][2] if pending else current_state
                pending.append(submit(next_index, initial))
                next_index += 1
            index, _, final_state, futures = pending.popleft()
            records = batches[index]
            tensors = [torch.from_numpy(future.result()) for future in futures]
            size = len(records)
            views = {view: torch.stack(tensors[i * size:(i + 1) * size])
                     for i, view in enumerate(CANONICAL_VIEWS)}
            torch.set_rng_state(final_state)
            yield records, views
    finally:
        for _, _, _, futures in pending:
            for future in futures:
                future.cancel()
