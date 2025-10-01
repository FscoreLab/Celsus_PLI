import numpy as np
import pandas as pd

def get_bbox_3d(img: np.ndarray):
    if not np.any(img):
        raise ValueError("Input array doesn't have nonzero elements")

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]] if r.any() else (0, 0)
    cmin, cmax = np.where(c)[0][[0, -1]] if c.any() else (0, 0)
    zmin, zmax = np.where(z)[0][[0, -1]] if z.any() else (0, 0)

    rmax += 1
    cmax += 1
    zmax += 1

    return rmin, cmin, zmin, rmax, cmax, zmax


def get_bbox_2d(img: np.ndarray):
    if np.any(img) is False:
        raise ValueError("Input array doesn't have nonzero elements")
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmax += 1
    cmax += 1

    return rmin, cmin, rmax, cmax


def expand_bbox(bbox: tuple[int, ...], shape: tuple[int, ...], expand_ratio: float):
    """
    Expands the bounding box while ensuring it remains within the array boundaries.

    :param bbox: tuple - Bounding box coordinates
                  (for 2D: (min_y, min_x, max_y, max_x),
                   for 3D: (min_z, min_y, min_x, max_z, max_y, max_x)).
    :param shape: tuple - Shape of the array
                   (for 2D: (height, width),
                    for 3D: (depth, height, width)).
    :param expand_ratio: float - Expansion factor (0.0 means no change, 1.0 doubles the size).
    :return: tuple - Adjusted bounding box within valid limits.
    """
    if not (0.0 <= expand_ratio <= 1.0):
        raise ValueError("expand_ratio must be in the range [0, 1]")

    dim = len(bbox) // 2
    expanded_bbox = [0] * len(bbox)

    for i in range(dim):
        min_val, max_val = bbox[i], bbox[i + dim]
        delta = int((max_val - min_val) * expand_ratio / 2)

        min_exp = max(0, min_val - delta)
        max_exp = min(shape[i], max_val + delta)

        # write to corresponding indices
        expanded_bbox[i] = min_exp
        expanded_bbox[i + dim] = max_exp

    return tuple(expanded_bbox)


def get_bbox_slices(mask: np.ndarray, dim: int, expand_ratio: float = 0.0) -> tuple[slice, ...]:
    """
    Retrieves bounding box slices for 2D or 3D masks and allows expansion by a given ratio.

    :param mask: (np.ndarray) - Binary mask to compute the bounding box.
    :param dim: int - Dimension of the bounding box (2 or 3).
    :param expand_ratio: float - Expansion factor (0.0 means no change, 1.0 doubles the size).
    :return: tuple - Tuple of slice objects representing the bounding box.
    """
    if dim == 3:
        bbox = get_bbox_3d(mask > 0)  # (min_z, min_y, min_x, max_z, max_y, max_x)
        if expand_ratio > 0:
            bbox = expand_bbox(bbox, mask.shape, expand_ratio)
        return tuple([slice(bbox[0], bbox[3]), slice(bbox[1], bbox[4]), slice(bbox[2], bbox[5])])

    elif dim == 2:
        bbox = get_bbox_2d(mask > 0)  # (min_y, min_x, max_y, max_x)
        if expand_ratio > 0:
            bbox = expand_bbox(bbox, mask.shape, expand_ratio)
        return tuple([slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])])

    else:
        raise ValueError("Only 2D & 3D bounding boxes are supported!")


def speedup_numpy_unique_cpu(array: np.ndarray, return_counts: bool = False):
    """
    Alternative accelerated version of numpy.unique for integer-based arrays
    """
    if return_counts is True:
        if not issubclass(array.dtype.type, np.integer):
            raise TypeError("The array must contain integers.")
        if np.any(array < 0):
            raise ValueError("The array must contain only non-negative integers.")
        counts = np.bincount(array.ravel())
        unique = np.where(counts != 0)[0]
        counts = counts[counts != 0]
        return unique, counts
    else:
        unique = np.sort(pd.unique(array.ravel()))
        return unique