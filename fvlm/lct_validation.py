import logging
from typing import Optional
import fastapi
import enum

import numpy as np

class BusinessErrorCode(enum.Enum):
    SERIES_ERROR = 1
    BODY_PART_ERROR = 2


class ServiceClientException(fastapi.HTTPException):
    """Base class for Complex Service response exceptions

    All responses return 422 Unprocessable Entity
    """

    def __init__(self, business_code: BusinessErrorCode, message: str) -> None:
        detail = {"type": business_code.name, "subtype": self.__class__.__name__, "message": message}
        super().__init__(status_code=422, detail=detail)


class SeriesError(ServiceClientException):
    """Base class for invalid series errors"""

    def __init__(self, message: str) -> None:
        super().__init__(business_code=BusinessErrorCode.SERIES_ERROR, message=message)

class BodyPartError(ServiceClientException):
    """Base class for invalid body part errors"""

    def __init__(self, message: str) -> None:
        super().__init__(business_code=BusinessErrorCode.BODY_PART_ERROR, message=message)

class LungsNotFoundError(BodyPartError):
    def __init__(self) -> None:
        message = (
            "Lungs not found on image or have too low relative percentage."
            + " Check that pixel intensities in the volume are in Housfield units."
        )
        super().__init__(message=message)

class InvalidLungsLengthError(SeriesError):
    def __init__(self, lung_length_mm: float):
        message = f"Lungs length is {lung_length_mm:.2f} mm which is too small"
        super().__init__(message=message)


def longest_segment(array: np.ndarray) -> int:
    """Find longest continous ones segment in the 0/1 array

    Parameters
    ----------
    array
        1D binary array

    Returns
    -------
        Length of the segment, 0 if no such segment
    """
    if (array == 0).all():
        return 0

    diff = np.diff(array)
    starts = (np.where(diff == 1)[0]).tolist()
    ends = (np.where(diff == -1)[0]).tolist()

    # Fix ones starting and ending the mask
    if array[0] == 1:
        starts.insert(0, 0)
    if array[-1] == 1:
        ends.append(array.size)

    return max([(end - start) for start, end in zip(starts, ends)])


class LungValidator:
    """Check chest CT image validity.

    Test CT image with following rules:
    1. CT is actually a chest CT
    2. Lungs are (almost) fully present

    Two features are used to classify CT image:
    1. lung mask length in mm
    2. lung mask axial occupance ratio


    Parameters
    ----------
        raises
            If validation doesn't hold, raise exception
    """

    OCCUPANCE_THRESHOLD = 0.35
    LUNG_LENGTH_THRESHOLD_MM = 145
    MIN_LUNG_PIXELS_ON_SLICE = 1500  # Critical for COVID/Cancer segmentors

    def __init__(self, raises=True, logger: Optional[logging.Logger] = None):
        self.raises = raises
        self.logger = logger or logging.getLogger(__name__)

    def validate_chest(self, lung_mask, z_spacing: float) -> bool:
        """Check that CT image is a chest CT based on lung occupance ratio

        Parameters
        ----------
        lung_mask
            Integer 3D lung mask with shape (depth, width, height)
        z_spacing
            Spacing in mm

        Returns
        -------
        lung_mask is a chest CT image

        Raises
        ------
        LungsNotFoundError
        """
        lung_length, lung_axial_occupancy_ratio = self._get_features(lung_mask, z_spacing)
        if lung_axial_occupancy_ratio < self.OCCUPANCE_THRESHOLD:
            if self.raises:
                raise LungsNotFoundError
            self.logger.warning(f"Not chest CT: {lung_axial_occupancy_ratio=}")
            return False
        self.logger.info(
            f"Lung occupancy ratio {lung_axial_occupancy_ratio:.3f}",
            extra={"props": {"occupancy": lung_axial_occupancy_ratio}},
        )
        return True

    def validate_lung_length(self, lung_mask, z_spacing: float) -> bool:
        """Check that lungs are fully present on the CT image

        Parameters
        ----------
        lung_mask
            Integer 3D lung mask with shape (depth, width, height)
        z_spacing
            Spacing in mm

        Returns
        -------
        lungs length is ok for a CT scan

        Raises
        ------
        InvalidLungsLengthError
        """
        lung_length, lung_axial_occupance_ratio = self._get_features(lung_mask, z_spacing)
        if lung_length < self.LUNG_LENGTH_THRESHOLD_MM:
            if self.raises:
                raise InvalidLungsLengthError(lung_length)
            self.logger.warning(f"Short lungs length: {lung_length=}")
            return False
        self.logger.info(f"Lung length {lung_length:.3f} mm", extra={"props": {"lung_length_mm": lung_length}})
        return True

    @staticmethod
    def _get_features(lung_mask: np.ndarray, z_spacing: float) -> tuple[float, float]:
        """Calculate features for CT image classification

        Parameters
        ----------
        lung_mask
            Integer 3D lung mask with shape (depth, width, height)

        Returns
        -------
            feature1
                lung mask length in mm
            feature2
                lung mask axial occupance ratio
        """
        # TODO: ImagePreprocessing.select_slices_lungs shares slice with lungs selection logic
        # Move it outside or resolve import order
        lung_pixels_per_slice = np.sum(lung_mask.astype(bool), axis=(1, 2))
        lung_axial_mask = (lung_pixels_per_slice > LungValidator.MIN_LUNG_PIXELS_ON_SLICE).astype(int)
        lung_length = longest_segment(lung_axial_mask)

        lung_length_mm = lung_length * z_spacing
        lung_occupancy_ratio = lung_length / lung_axial_mask.size

        return lung_length_mm, lung_occupancy_ratio