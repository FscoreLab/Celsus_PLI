import cv2
import numpy as np
import torch
from skimage import morphology
from sklearn.cluster import KMeans
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.transforms import CropForegroundd, CropForeground
# import functools
# import matplotlib.pyplot as plt
# from monai.transforms import CropForeground
# from kate.businesslogic import api_helpers



class CropForegrounddInference(CropForegroundd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lazy=False, **kwargs)  # CropForeground incorrectly work with lazy=True
        self.inverse_inference = True

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropForeground
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        ## My changes
        lazy_ = self.lazy  # if lazy is None else lazy
        ## --END--
        for key, m in self.key_iterator(d, self.mode):
            if d[key] is not None:
                d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d

    def inverse(self, data: Mapping) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        d = dict(data)
        orig_xy_size = d["image_meta_dict"]["orig_xy_size"]
        fg_start_coord = d["image_meta_dict"]["foreground_start_coord"]
        fg_end_coord = d["image_meta_dict"]["foreground_end_coord"]
        for key in d:  # d[key] - HWD
            if isinstance(d[key], np.ndarray):
                shape = d[key].shape
                restored_arr = np.zeros((shape[0], *orig_xy_size[::-1], shape[-1]), dtype=d[key].dtype)
                restored_arr[
                    :,
                    fg_start_coord[0] : fg_end_coord[0],
                    fg_start_coord[1] : fg_end_coord[1],
                    fg_start_coord[2] : fg_end_coord[2],
                ] = d[key]
                d[key] = restored_arr
            if isinstance(d[key], torch.Tensor):
                shape = d[key].shape
                restored_arr = torch.zeros(
                    (shape[0], *orig_xy_size[::-1], shape[-1]), dtype=d[key].dtype, device=d[key].device
                )
                restored_arr[
                    :,
                    fg_start_coord[0] : fg_end_coord[0],
                    fg_start_coord[1] : fg_end_coord[1],
                    fg_start_coord[2] : fg_end_coord[2],
                ] = d[key]
                d[key] = restored_arr

        return d

def apply_kmeans_thresholding(image: np.ndarray, n_clusters: int = 2) -> float:
    """
    Apply KMeans clustering to find a threshold value for the given image.
    """
    reshaped_img = np.reshape(image, [np.prod(image.shape), 1])
    kmeans = KMeans(n_clusters=n_clusters).fit(reshaped_img)
    centers = sorted(kmeans.cluster_centers_.flatten())
    return np.mean(centers)


def apply_morphological_operations(image: np.ndarray, erosion_size: int, dilation_size: int) -> np.ndarray:
    """
    Apply erosion and dilation to the image.
    """
    dilated = morphology.dilation(image, np.ones([dilation_size, dilation_size]))
    return morphology.erosion(dilated, np.ones([erosion_size, erosion_size]))


def find_foreground_bbox_2d(
    rescaled_img: np.ndarray, erosion_size: int = 2, dilation_size: int = 1, border_size: int = 25
) -> tuple:
    """
    Find foreground bbox for image using morphological operations, KMeans clustering and thresholding.

    Parameters
    ----------
    rescaled_img : np.ndarray
        Rescaled 2d image
    erosion_size : int
        Kernel size for applying erosion
    dilation_size : int
        Kernel size for applying dilation
    border_size : int
        Border size for expanding image before processing

    Returns
    -------
    Bbox and flag telling whether the image was successfully processed or not
    """
    try:
        orig_size = np.prod(rescaled_img.shape)

        img_with_border = cv2.copyMakeBorder(
            rescaled_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0
        )

        row_size, col_size = img_with_border.shape
        middle_section = img_with_border[
            int(col_size / 5) : int(col_size / 5 * 4), int(row_size / 5) : int(row_size / 5 * 4)
        ]
        threshold = apply_kmeans_thresholding(middle_section)

        # Find foreground mask
        thresholded_img = np.where(img_with_border > threshold, 1.0, 0.0)
        foreground = apply_morphological_operations(thresholded_img, erosion_size, dilation_size).astype("uint8")

        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)

        foreground_largest = np.zeros(img_with_border.shape, np.uint8)
        cv2.drawContours(foreground_largest, [largest_contour], -1, 255, cv2.FILLED)

        # Filling the holes in the enclosed mask
        mask = np.zeros((row_size + 2, col_size + 2), np.uint8)
        cv2.floodFill(foreground_largest, mask, (0, 0), 1)

        # Applying the mask on the image
        masked_img = (img_with_border * 255).astype("uint8")
        masked_img = (1 - mask[:row_size, :col_size]) * masked_img

        # Cropping the masked image
        _, thresh = cv2.threshold(masked_img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            x, y = x - border_size, y - border_size
            bbox_coords = [x, y, x + w, y + h]
            cropping_error = w * h < int(
                orig_size / 27
            )  # For example, let orig_size=512x512 then if crop_size=90*90 we won't crop image
        else:
            bbox_coords = [0, 0, rescaled_img.shape[1], rescaled_img.shape[0]]
            cropping_error = True
    except Exception as e:
        print(f"Catched exception {e} for find_foreground_bbox_2d function!")
        bbox_coords = [0, 0, rescaled_img.shape[1], rescaled_img.shape[0]]
        cropping_error = True

    return bbox_coords, cropping_error


def get_union_2d_bbox(bboxes):
    x_min_union = min(bbox[0] for bbox in bboxes)
    y_min_union = min(bbox[1] for bbox in bboxes)
    x_max_union = max(bbox[2] for bbox in bboxes)
    y_max_union = max(bbox[3] for bbox in bboxes)

    return [x_min_union, y_min_union, x_max_union, y_max_union]


def find_foreground_3d_mask_by_3_slices(
    rescaled_img: np.ndarray, erosion_size: int = 2, dilation_size: int = 4
) -> tuple:
    """
    Find foreground 3D mask for 3D image using three slices, morphological operations, KMeans clustering and thresholding.
    """
    size = rescaled_img.shape[0]
    first_foreground_bbox_coords, first_cropping_error = find_foreground_bbox_2d(
        rescaled_img[int(size * 0.35)], erosion_size=erosion_size, dilation_size=dilation_size
    )
    second_foreground_bbox_coords, second_cropping_error = find_foreground_bbox_2d(
        rescaled_img[int(size * 0.5)], erosion_size=erosion_size, dilation_size=dilation_size
    )
    third_foreground_bbox_coords, third_cropping_error = find_foreground_bbox_2d(
        rescaled_img[int(size * 0.65)], erosion_size=erosion_size, dilation_size=dilation_size
    )
    cropping_error = all([first_cropping_error, second_cropping_error, third_cropping_error])

    if cropping_error:
        return np.ones(rescaled_img.shape, dtype=bool)
    else:
        union_bbox = get_union_2d_bbox(
            [first_foreground_bbox_coords, second_foreground_bbox_coords, third_foreground_bbox_coords]
        )
        x_min, y_min, x_max, y_max = union_bbox
        foreground_bbox_mask = np.zeros(rescaled_img.shape, bool)
        foreground_bbox_mask[:, y_min:y_max, x_min:x_max] = True
        return foreground_bbox_mask


def rescale_window(arr, lower=-1250, upper=150):
    out_arr = torch.clip(arr, min=lower, max=upper)
    out_arr = (out_arr - lower) / (upper - lower)

    return out_arr


def find_foreground_3d_mask_by_3_slices_torch(
    img: torch.Tensor, erosion_size: int = 2, dilation_size: int = 4
) -> tuple:
    """
    Find foreground 3D mask for 3D image using three slices, morphological operations, KMeans clustering and thresholding.
    """
    rescaled_img = rescale_window(img, lower=-1250, upper=200)
    rescaled_img = rescaled_img.cpu().numpy()[0]
    rescaled_img = np.moveaxis(rescaled_img, -1, 0)

    foreground_bbox_mask = find_foreground_3d_mask_by_3_slices(
        rescaled_img, erosion_size=erosion_size, dilation_size=dilation_size
    )
    foreground_bbox_mask = np.moveaxis(foreground_bbox_mask, 0, -1)
    foreground_bbox_mask = torch.from_numpy(foreground_bbox_mask[None, ...]).to(img.device)
    return foreground_bbox_mask


# if __name__ == "__main__":
#     file = "/astral-data/Minio/kate/data/interim/temp_crop_roi_fuck_packs/6"
#     array_no_rescale, preprocessor = api_helpers.initial_preprocessing(file, change_size=False, resample=False)
#
#     out = preprocessor.forward(array_no_rescale)
#     array, array_no_rescale = out
#
#     cropper = CropForeground(
#         select_fn=functools.partial(find_foreground_3d_mask_by_3_slices, erosion_size=7, dilation_size=2), margin=0
#     )
#     crop = cropper(array)
#
#     n = int(array.shape[0] / 2)
#     plt.subplot(1, 2, 1)
#     plt.imshow(array[n], cmap="gray")
#     plt.subplot(1, 2, 2)
#     plt.imshow(crop[n], cmap="gray")
#     plt.show()
