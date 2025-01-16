import math
import numpy as np
from scipy.interpolate import interp1d


def remap_to_polar(image):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    max_radius = int(np.sqrt(center[0]**2 + center[1]**2))

    polar_image = np.zeros((max_radius, 360), dtype=np.uint8)

    for angle in range(360):
        for radius in range(max_radius):
            x = round(radius * np.cos(np.deg2rad(angle))) + center[0]
            y = round(radius * np.sin(np.deg2rad(angle))) + center[1]
            if 0 <= x < width and 0 <= y < height:
                polar_image[radius, angle] = image[y, x]

    return polar_image


def remap_to_cartesian(polar_image, size):
    max_radius, num_angles = polar_image.shape[:2]
    center = (size // 2, size // 2)

    cartesian_image = np.zeros((size, size), dtype=np.uint16)

    for x in range(size):
        for y in range(size):
            radius = round(np.sqrt((x-center[0])**2 + (y-center[1])**2))
            angle = round(math.degrees(math.atan2(y-center[1], x-center[0])))
            angle = (angle + 360) % 360  # Ensure angle is within [0, 360)
            if 0 <= angle < num_angles and 0 <= radius < max_radius:
                cartesian_image[y, x] = polar_image[radius, angle]

    return cartesian_image


def resize_array_nearest(array, new_size):
    old_size = len(array)
    x_old = np.arange(old_size, dtype=array.dtype)
    f = interp1d(x_old, array, kind='nearest', fill_value='extrapolate')
    x_new = np.linspace(0, old_size - 1, new_size)
    return f(x_new)


def extend_myo_to_whole_height(image_array, mask_array, height = None):
    image_array = image_array.copy()
    image_array[mask_array == 0] = 0
    width = image_array.shape[1]
    if height is None:
        height = image_array.shape[0]
    myo_image_array_expanded = np.zeros((height, width), image_array.dtype)
    for i in range(image_array.shape[1]):
        column = image_array[:, i]
        column_trim = np.trim_zeros(column)
        if np.any(column_trim != 0):
            new_column = resize_array_nearest(column_trim, height)
            myo_image_array_expanded[:, i] = new_column
    return myo_image_array_expanded


def back_myo_to_origin_shape(image_array_expanded, original_mask):
    myo_image_array_expanded_back = np.zeros(original_mask.shape, image_array_expanded.dtype)
    for i in range(image_array_expanded.shape[1]):
        column_expanded = image_array_expanded[:, i]
        column = original_mask[:, i]
        if np.any(column != 0):
            start = np.nonzero(column)[0][0]
            end = np.nonzero(column)[0][-1]
            column_trim = np.trim_zeros(column)
            new_column = resize_array_nearest(column_expanded, len(column_trim))
            myo_image_array_expanded_back[start:end+1, i] = new_column
    return myo_image_array_expanded_back
