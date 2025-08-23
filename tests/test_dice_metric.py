import torch

from segmentation.helpers.dice_metric import DiceMetric


def test_multi_dice_binary_perfect_match():
    # Single-sample, 2 classes, 2x2 image
    # Ground truth: class 1 everywhere
    targs = torch.ones((1, 2, 2), dtype=torch.long)
    # Logits/preds: channel 1 has higher score everywhere
    preds = torch.tensor(
        [[[[0.1, 0.1], [0.1, 0.1]], [[0.9, 0.9], [0.9, 0.9]]]]
    )

    metric = DiceMetric(1e-6)
    dice = metric.multi_dice(preds, targs, class_id=1)
    assert torch.isclose(dice, torch.tensor(1.0), atol=1e-6)


def test_multi_dice_partial_overlap():
    # Ground truth: top-left and bottom-right are class 1
    targs = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.long)
    # Predictions: shift one pixel (only one overlap)
    preds = torch.tensor(
        [[[[0.2, 0.8], [0.8, 0.2]], [[0.8, 0.2], [0.2, 0.8]]]]
    )

    metric = DiceMetric(1e-6)
    dice = metric.multi_dice(preds, targs, class_id=1)
    # One TP, three non-positives -> Dice = 2*1 / (1+3) = 0.5 (with smoothing ~0.5)
    assert torch.isclose(dice, torch.tensor(0.5), atol=1e-2)

