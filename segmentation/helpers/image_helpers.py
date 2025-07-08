
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import F
from PIL import Image


def cut_to_square(image_array):
    width, height = image_array.shape

    min_row = 0
    max_row = height
    min_col = 0
    max_col = width

    if width > height:
        delta = int((width - height) / 2)
        min_row = delta
        max_row = width - delta
    elif height > width:
        delta = int((height - width) / 2)
        min_col = delta
        max_col = height - delta

    return image_array[min_row:max_row, min_col:max_col]

def get_mask_bounding_box(mask, expansion_percent = 0.0, square=True):
    nonzero_pixels = np.nonzero(mask)

    min_row = np.min(nonzero_pixels[0])
    max_row = np.max(nonzero_pixels[0])
    min_col = np.min(nonzero_pixels[1])
    max_col = np.max(nonzero_pixels[1])

    width = max_col - min_col
    height = max_row - min_row

    if(square):
        max_side = max(width, height)

        if(width > height):
            min_row -= int((width - height) / 2)
            max_row = min_row + max_side
        else:
            min_col -= int((height - width) / 2)
            max_col = min_col + max_side

        width = max_side
        height = max_side

    expansion_row_number = int(expansion_percent * width / 2)
    expansion_col_number = int(expansion_percent * height / 2)

    if expansion_percent > 0:
        expansion_row_number = max(1, expansion_row_number)
        expansion_col_number = max(1, expansion_col_number)

    return (
        int(min_row - expansion_row_number),
        int(max_row + expansion_row_number),
        int(min_col - expansion_col_number),
        int(max_col + expansion_col_number),
        int(width + expansion_row_number * 2),
        int(height + expansion_col_number * 2)
    )

def normalize_image(image_array, min_value=None, max_value=None):
    if min_value is None:
        min_value = np.min(image_array)
    if max_value is None:
        max_value = np.max(image_array)

    image_array[image_array < min_value] = min_value
    image_array[image_array > max_value] = max_value
    return ((image_array - min_value) / (max_value - min_value)) * 255

def remap_mask(mask_array, original_classes, target_classes):
    assert len(original_classes) == len(target_classes), 'List of target and original classes should have the same length'

    for original, target in zip(original_classes, target_classes):
        mask_array[mask_array == original] = target
    return mask_array

def interpolate_mask(mask, size, mode='nearest'):
    return F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=size,
        mode=mode
    ).squeeze().byte()

def scale_numbers(array):
    min_value = min(array)
    max_value = max(array)

    new_min = 0
    new_max = 255

    return [int((x - min_value) / (max_value - min_value) * (new_max - new_min) + new_min) for x in array]


def make_transparent_background_for_mask(mask, cmap_name = 'inferno'):
    cmap = plt.get_cmap(cmap_name)
    mask_array = np.array(mask)

    # Find unique classes in the mask
    unique_classes = np.unique(mask_array)
    rgba_data = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)

    if unique_classes.max() != 0:
        for mask_class, color in zip(unique_classes, scale_numbers(unique_classes)):
            if mask_class == 0:
                continue
            class_color = [int(c * 255) for c in cmap(color)]

            for i in range(len(mask_array)):
                for j in range(len(mask_array[i])):
                    if mask_array[i][j] == mask_class:
                        rgba_data[i][j] = class_color

    return Image.fromarray(rgba_data, 'RGBA')


def show_image_and_mask(
        image_array,
        mask_array,
        mask_alpha=0.5,
        image_cmap='gray',
        mask_cmap='inferno',
        image_title='Image',
        mask_title='Mask',
        size=(12, 6)
    ):
    mask_tr_img = make_transparent_background_for_mask(mask_array)

    _, axs = plt.subplots(1, 2, figsize=size)
    axs[0].set_title(image_title)
    axs[0].imshow(image_array, cmap=image_cmap)
    axs[1].set_title(mask_title)
    axs[1].imshow(image_array, cmap=image_cmap)
    axs[1].imshow(mask_tr_img, alpha=mask_alpha, cmap=mask_cmap)
    plt.show()

def show_image(
        image_array,
        cmap='gray',
        title='Mask',
        size=(6, 6)
    ):
    _, axs = plt.subplots(1, 1, figsize=size)
    axs.set_title(title)
    axs.imshow(image_array, cmap=cmap)
    plt.show()
