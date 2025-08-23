from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import segmentation.learner as seg_learner
from segmentation.learner import Learner


class DummyLearner:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    # No-op training hooks
    def fit_one_cycle(self, *args, **kwargs):
        return self

    def unfreeze(self):
        return self

    # For API compatibility in export tests elsewhere
    def export(self, *args, **kwargs):
        return None


def _write_png(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_learner_builds_dataloaders_with_synthetic_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create tiny synthetic dataset
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"

    img = (np.ones((16, 16), dtype=np.uint8) * 128)
    msk = np.zeros((16, 16), dtype=np.uint8)
    msk[4:12, 4:12] = 1  # foreground square

    _write_png(images_dir / "sample_01.png", img)
    _write_png(masks_dir / "sample_01.png", msk)

    # Monkeypatch unet_learner to avoid heavy model creation
    def _fake_unet_learner(dataloaders, *args, **kwargs):
        return DummyLearner(dataloaders)

    monkeypatch.setattr(seg_learner, "unet_learner", _fake_unet_learner)

    config = {
        "classes": ["bg", "fg"],
        "size": 16,
        "dataset_path": str(tmp_path),
        "batch_size": 1,
        "num_workers": 0,
        "architecture": None,
        "epoch_number": 1,
        "model_output_path": str(tmp_path / "out"),
        "model_name": "test",
    }

    learner = Learner(config)
    model = learner.learn()

    # Assert dataloaders are non-empty and iterable
    dls = model.dataloaders
    # train and valid should each have at least one batch
    assert len(dls.train) >= 1
    assert len(dls.valid) >= 1

