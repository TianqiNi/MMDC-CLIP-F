"""Exact preprocessing/RNG equivalence, including stochastic consumers and resume."""

import copy

import numpy as np
import pytest
import torch
from PIL import Image

from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.production import role_image_batch_loader
from mmdc_clip_f.research.view_risk.roles import PrivateExamRecord, Role, ViewReference
from mmdc_clip_f.research.view_risk.training import FrozenClassifierSchedule


@pytest.fixture
def images(tmp_path):
    records = []
    for i in range(5):
        views = {}
        for j, view in enumerate(CANONICAL_VIEWS):
            path = tmp_path / f"{i}-{view}.png"
            pixels = np.random.default_rng(i * 4 + j).integers(
                0, 256, (31 + i * 5, 41 + j * 7, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(path)
            views[view] = ViewReference(path.stem, path.name, sha256_file(path))
        records.append(PrivateExamRecord(
            "RSNA", f"exam-{i}", f"patient-{i}", i % 4, "fixture", views,
            Role.CLASSIFIER_FIT,
        ))
    return tmp_path, tuple(records)


def loader(images, workers, epoch=1, training=True):
    root, records = images
    schedule = FrozenClassifierSchedule()
    return role_image_batch_loader(
        records, image_root=root, image_size=24, optimizer=schedule.optimizer,
        augmentation=schedule.augmentation, epoch=epoch, seed=42, training=training,
        image_workers=workers, prefetch_batches=2,
    )


@pytest.mark.parametrize("training,interleave", [(True, False), (True, True), (False, True)])
def test_parallel_pixels_order_labels_and_rng_match_serial(images, training, interleave):
    def collect(workers):
        torch.manual_seed(932)
        result = []
        for epoch in (1, 2):
            for batch in loader(images, workers, epoch, training):
                result.append((batch, torch.get_rng_state().clone()))
                if interleave:
                    # CPU dropout or another stochastic consumer between batches.
                    torch.rand(17)
        return result, torch.get_rng_state()

    serial, serial_rng = collect(0)
    parallel, parallel_rng = collect(2)
    assert [len(b.exam_keys) for b, _ in parallel] == [3, 2, 3, 2]
    for (a, a_rng), (b, b_rng) in zip(serial, parallel, strict=True):
        assert a.exam_keys == b.exam_keys
        assert torch.equal(a.payload.labels, b.payload.labels)
        for view in CANONICAL_VIEWS:
            assert torch.equal(a.payload.views[view], b.payload.views[view])
        assert torch.equal(a_rng, b_rng)
    assert torch.equal(serial_rng, parallel_rng)


def test_parallel_adam_updates_and_epoch_resume_match_serial(images):
    def run(workers, resume=None, epochs=(1, 2)):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(12, 8), torch.nn.Dropout(0.3),
                                    torch.nn.Linear(8, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if resume is not None:
            model.load_state_dict(resume[0])
            optimizer.load_state_dict(resume[1])
            torch.set_rng_state(resume[2])
        for epoch in epochs:
            for batch in loader(images, workers, epoch):
                features = torch.cat([batch.payload.views[v].mean((2, 3))
                                      for v in CANONICAL_VIEWS], dim=1)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(features), batch.payload.labels)
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), \
            torch.get_rng_state()

    expected = run(0)
    actual = run(2, resume=run(0, epochs=(1,)), epochs=(2,))
    for name in expected[0]:
        assert torch.equal(expected[0][name], actual[0][name])
    for key, state in expected[1]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, actual[1]["state"][key][name])
    assert torch.equal(expected[2], actual[2])


def test_parallel_reader_still_rejects_changed_image_bytes(images):
    root, records = images
    (root / records[0].views["L_CC"].path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        list(loader(images, 2, training=False))
