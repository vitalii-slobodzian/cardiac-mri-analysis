import numpy as np
import torch
from PIL import Image

from segmentation.helpers.image_helpers import (
    normalize_image,
    get_mask_bounding_box,
    remap_mask,
    interpolate_mask,
    make_transparent_background_for_mask,
)
from segmentation.helpers.cut_to_square import CutToSquare


def test_normalize_image_linear_scale():
    arr = np.array([[0, 5], [10, 10]], dtype=float)
    out = normalize_image(arr.copy(), min_value=0, max_value=10)
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 255.0)


def test_get_mask_bounding_box_square():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 4:6] = 1  # height=4, width=2 -> square expands width
    r0, r1, c0, c1, w, h = get_mask_bounding_box(mask, expansion_percent=0.0, square=True)
    # Expect square box centered on the mask with max side 4
    assert (r1 - r0) == (c1 - c0) == max(4, 2)
    assert w == h == max(4, 2)


def test_remap_mask_simple():
    mask = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    out = remap_mask(mask.copy(), original_classes=[0, 1, 2], target_classes=[0, 2, 1])
    assert set(np.unique(out)) == {0, 1, 2}
    # class 1 -> 2, class 2 -> 1
    assert out[0, 1] == 2 and out[0, 2] == 1


def test_interpolate_mask_preserves_values_and_shape():
    mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)
    out = interpolate_mask(mask, size=(4, 4), mode='nearest')
    assert out.shape == torch.Size([4, 4])
    assert set(torch.unique(out).tolist()) <= {0, 1}


def test_make_transparent_background_for_mask_returns_rgba_image():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 2
    img = make_transparent_background_for_mask(mask)
    assert isinstance(img, Image.Image)
    assert img.mode == 'RGBA' and img.size == (5, 5)


def test_cut_to_square_transform_tuple_returns_pil_types():
    image = np.ones((4, 4), dtype=np.uint8) * 128
    mask = np.zeros((4, 4), dtype=np.uint8)
    t = CutToSquare()
    pil_img, pil_mask = t.encodes((image, mask))
    assert hasattr(pil_img, 'to_thumb')  # Fastai PILImage
    assert hasattr(pil_mask, 'to_thumb')  # Fastai PILMask

