"""Inference utilities for the 3D MedNeXt classifier.

This module provides two high-level entry points:

``init_model``
    Loads the trained classifier with the preprocessing pipeline that mirrors
    validation-time transforms. All parameters required to restore the model
    (weights, architecture, preprocessing settings) are hard-coded via
    ``DEFAULT_*`` constants so that downstream entry points can be used without
    additional configuration.

``run_inference``
    Consumes either a single ``nii.gz`` file or a CSV manifest following the
    training format. In the CSV case, the function iterates through the split
    using a MONAI ``DataLoader`` and writes batched predictions to the provided
    CSV path, supporting resumable execution by skipping already processed
    studies.

The helper ``predict_single_volume`` is exposed for convenience when only a
single scan needs to be evaluated, for example when powering an interactive
demo.
"""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    CastToTypeD,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    SpatialPadd,
    Spacingd,
)
from torch import nn

from mednext.crop_human_body import (
    CropForegrounddInference,
    find_foreground_3d_mask_by_3_slices_torch,
)
from mednext.mednext_classifier3d import MedNeXt3DClassifier

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded model artefact locations and defaults.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[0] / "artifacts"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_DIR / "pli_2.json"
DEFAULT_WEIGHTS_PATH = DEFAULT_MODEL_DIR / "pli_2_model_last_e_15.ckpt"


@dataclass
class InferenceArtifacts:
    """Container with all objects needed to perform inference."""

    model: nn.Module
    device: torch.device
    classes: Sequence[str]
    transform: Compose
    image_column: str
    id_column: Optional[str]
    roi_size: Tuple[int, int, int]
    crop_by_body_part: bool
    config: Dict[str, object]
    amp_enabled: bool
    amp_dtype: torch.dtype
    input_dtype: torch.dtype
    probability_activation: str


class _DicomZipConversion:
    """Helper responsible for expanding and converting a DICOM ZIP archive."""

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        self._extract_dir: Optional[tempfile.TemporaryDirectory] = None
        self._output_dir: Optional[tempfile.TemporaryDirectory] = None
        self._nifti_path: Optional[Path] = None

    def convert(self) -> Path:
        if self._nifti_path is not None:
            return self._nifti_path

        if not self.zip_path.exists():
            raise FileNotFoundError(f"DICOM ZIP archive not found: {self.zip_path}")

        self._extract_dir = tempfile.TemporaryDirectory(prefix="dicom_zip_extract_")
        extract_root = Path(self._extract_dir.name)
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extractall(extract_root)

        self._output_dir = tempfile.TemporaryDirectory(prefix="dicom_zip_nifti_")
        output_path = Path(self._output_dir.name) / f"{self.zip_path.stem}.nii.gz"
        self._write_nifti(extract_root, output_path)
        self._nifti_path = output_path
        return self._nifti_path

    def cleanup(self) -> None:
        if self._output_dir is not None:
            self._output_dir.cleanup()
            self._output_dir = None
        if self._extract_dir is not None:
            self._extract_dir.cleanup()
            self._extract_dir = None

    def _write_nifti(self, dicom_root: Path, output_path: Path) -> None:
        try:  # pragma: no cover - import guard for optional dependency
            import SimpleITK as sitk
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "SimpleITK is required to convert DICOM ZIP archives to NIfTI. "
                "Install it with `pip install SimpleITK`."
            ) from exc

        candidate_dirs = [dicom_root]
        candidate_dirs.extend(sorted(p for p in dicom_root.rglob("*") if p.is_dir()))
        for directory in candidate_dirs:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(directory))
            if not series_ids:
                continue
            series_id = series_ids[0]
            file_names = reader.GetGDCMSeriesFileNames(str(directory), series_id)
            if not file_names:
                continue
            reader.SetFileNames(file_names)
            image = reader.Execute()
            sitk.WriteImage(image, str(output_path))
            return

        raise ValueError(
            f"No readable DICOM series found inside archive {self.zip_path}. "
            "Ensure the ZIP contains a single series."
        )


class DicomZipCache:
    """Cache that keeps temporary conversions alive for the duration of inference."""

    def __init__(self) -> None:
        self._conversions: Dict[str, _DicomZipConversion] = {}

    def resolve(self, zip_path: Path) -> Path:
        key = str(Path(zip_path).resolve())
        conversion = self._conversions.get(key)
        if conversion is None:
            conversion = _DicomZipConversion(Path(zip_path))
            conversion.convert()
            self._conversions[key] = conversion
        return conversion.convert()

    def cleanup(self) -> None:
        for conversion in self._conversions.values():
            conversion.cleanup()
        self._conversions.clear()


def _ensure_sequence_of_ints(value: Union[int, Sequence[int], str, None]) -> Tuple[int, ...]:
    """Return ``value`` as a tuple of ints, tolerating string encodings."""

    if value is None:
        return tuple()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        separators = (",", ";", " ")
        for sep in separators:
            value = value.replace(sep, "-")
        parts = [part for part in value.split("-") if part]
        return tuple(int(part) for part in parts)
    return tuple(int(v) for v in value)


def _resolve_torch_dtype(value: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    """Resolve ``value`` to a torch dtype, defaulting to ``torch.float16``."""

    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"float16", "half", "fp16"}:
            return torch.float16
        if normalised in {"float32", "float", "fp32"}:
            return torch.float32
        if normalised in {"bfloat16", "bf16"}:
            return torch.bfloat16
    return torch.float16


def _apply_probability_activation(logits: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply the configured probability activation function to ``logits``."""

    activation = activation.lower()
    if activation == "softmax":
        return torch.softmax(logits, dim=1)
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    raise ValueError(f"Unsupported probability activation: {activation}")


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    """Remove ``prefix`` from keys in ``state_dict`` if present."""

    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _amp_context(artifacts: InferenceArtifacts):
    """Return the appropriate autocast context manager for the current device."""

    if artifacts.amp_enabled and artifacts.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=artifacts.amp_dtype)
    return nullcontext()


def _is_nifti_file(path: Path) -> bool:
    suffix = "".join(path.suffixes).lower()
    return suffix.endswith(".nii") or suffix.endswith(".nii.gz")


def _is_dicom_zip(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def _create_inference_transforms(
    roi_size: Sequence[int], *, crop_by_body_part: bool = False, dtype: torch.dtype = torch.float16
) -> Compose:
    """Validation-time preprocessing pipeline used for inference."""

    roi_size = tuple(int(v) for v in roi_size)
    transforms: List = [
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",)),
        Spacingd(keys=("image",), pixdim=(1.0, 1.0, 1.5), mode="bilinear"),
        Lambdad(keys=("image",), func=lambda x: np.clip(x, -1000.0, 1000.0)),
    ]

    if crop_by_body_part:
        select_fn = partial(
            find_foreground_3d_mask_by_3_slices_torch, erosion_size=7, dilation_size=2
        )
        transforms.append(
            CropForegrounddInference(
                keys=["image"],
                source_key="image",
                select_fn=select_fn,
                margin=0,
                allow_missing_keys=True,
            )
        )

    transforms.extend(
        [
            SpatialPadd(keys=("image",), spatial_size=roi_size),
            CenterSpatialCropd(keys=("image",), roi_size=roi_size),
            NormalizeIntensityd(keys=("image",), nonzero=True, channel_wise=True),
            CastToTypeD(keys=("image",), dtype=dtype),
            EnsureTyped(keys=("image",), dtype=dtype),
        ]
    )
    return Compose(transforms)


def init_model(device: Optional[Union[str, torch.device]] = None) -> InferenceArtifacts:
    """Initialise the MedNeXt classifier with hard-coded artefact paths.

    Parameters
    ----------
    device:
        Optional device specifier. When omitted, CUDA is selected if available.

    Returns
    -------
    InferenceArtifacts
        Object bundling the model, preprocessing pipeline and metadata required
        for downstream inference utilities.
    """

    config_path = DEFAULT_CONFIG_PATH
    weights_path = DEFAULT_WEIGHTS_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration not found at {config_path}. Adjust DEFAULT_CONFIG_PATH if necessary."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Adjust DEFAULT_WEIGHTS_PATH if necessary."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, object] = json.load(f)

    classes = list(config.get("classes", []))
    if not classes:
        raise ValueError("Model configuration must define a non-empty 'classes' list")

    roi_size = tuple(int(v) for v in config.get("roi_size", (320, 320, 224)))
    crop_by_body_part = bool(config.get("crop_by_body_part", False))
    input_dtype = _resolve_torch_dtype(
        config.get("image_dtype")
        or config.get("inference_dtype")
        or config.get("input_dtype")
    )
    probability_activation = str(config.get("probability_activation", "sigmoid")).lower()
    if probability_activation not in {"sigmoid", "softmax"}:
        LOGGER.warning(
            "Unknown probability activation '%s'; defaulting to sigmoid",
            probability_activation,
        )
        probability_activation = "sigmoid"

    depth_config = _ensure_sequence_of_ints(config.get("depth_config"))
    if not depth_config:
        depth_config = (2, 2, 2, 2, 2)
    expansion_ratio = _ensure_sequence_of_ints(config.get("expansion_ratio"))
    if not expansion_ratio:
        expansion_ratio = (4, 4, 4, 4, 4)

    model = MedNeXt3DClassifier(
        in_channels=int(config.get("in_channels", 1)),
        num_classes=len(classes),
        base_channels=int(config.get("base_channels", 32)),
        depth_config=depth_config,
        exp_r=expansion_ratio,
        kernel_size=int(config.get("kernel_size", 3)),
        norm_type=str(config.get("norm_type", "group")),
        grn=bool(config.get("use_grn", False)),
        dropout=float(config.get("dropout", 0.0)),
    )

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    state_dict = _strip_state_dict_prefix(state_dict, prefix="module.")
    model.load_state_dict(state_dict, strict=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    amp_dtype = _resolve_torch_dtype(config.get("amp_dtype") or input_dtype)
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    if amp_dtype == torch.float16 and device.type != "cuda":
        # CUDA autocast supports float16; fallback to bfloat16 on other devices when requested.
        amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float32
    if device.type != "cuda":
        amp_enabled = False

    model.to(device)
    model.eval()

    transform = _create_inference_transforms(
        roi_size=roi_size,
        crop_by_body_part=crop_by_body_part,
        dtype=input_dtype,
    )

    return InferenceArtifacts(
        model=model,
        device=device,
        classes=classes,
        transform=transform,
        image_column=str(config.get("image_column", "study_file")),
        id_column=config.get("id_column"),
        roi_size=roi_size,
        crop_by_body_part=crop_by_body_part,
        config=config,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        input_dtype=input_dtype,
        probability_activation=probability_activation,
    )


def predict_single_volume(
    volume_path: Union[str, Path], artifacts: InferenceArtifacts
) -> Dict[str, float]:
    """Run the classifier on a single ``nii.gz`` volume."""

    path = Path(volume_path)
    if _is_dicom_zip(path):
        conversion = _DicomZipConversion(path)
        try:
            converted_path = conversion.convert()
            return predict_single_volume(converted_path, artifacts)
        finally:
            conversion.cleanup()

    if not _is_nifti_file(path):
        raise ValueError(
            f"Unsupported volume format '{path.suffix}'. Provide a NIfTI file or a DICOM ZIP archive."
        )

    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")

    data = {"image": str(path)}
    data = artifacts.transform(data)
    image = data["image"].unsqueeze(0).to(
        device=artifacts.device,
        dtype=artifacts.input_dtype,
        non_blocking=True,
    )

    with torch.no_grad():
        with _amp_context(artifacts):
            logits = artifacts.model(image)
            probs = _apply_probability_activation(logits, artifacts.probability_activation)
        probs = probs.squeeze(0).float().cpu().numpy()

    return {cls: float(prob) for cls, prob in zip(artifacts.classes, probs)}


def _resolve_volume_path(root: Optional[Path], value: Union[str, Path]) -> Path:
    """Resolve ``value`` against ``root`` if it is not absolute."""

    path = Path(value)
    if not path.is_absolute():
        if root is None:
            raise ValueError(
                "Relative image paths require a data root directory. Provide 'data_dir' when calling run_inference."
            )
        path = root / path
    return path.expanduser().resolve()


def _prepare_manifest_subset(
    df: pd.DataFrame,
    image_column: str,
    pending_indices: Iterable[int],
    *,
    data_root: Optional[Path],
    id_column: Optional[str],
    dicom_cache: Optional[DicomZipCache] = None,
) -> List[Dict[str, object]]:
    """Create MONAI-compatible dictionaries for the pending manifest rows."""

    records: List[Dict[str, object]] = []
    for idx in pending_indices:
        row = df.loc[idx]
        image_value = row[image_column]
        original_path = _resolve_volume_path(data_root, image_value)
        effective_path = original_path
        if _is_dicom_zip(original_path):
            if dicom_cache is None:
                raise ValueError(
                    "Encountered a DICOM ZIP archive but no conversion cache was provided."
                )
            effective_path = dicom_cache.resolve(original_path)
        elif not _is_nifti_file(original_path):
            raise ValueError(
                f"Unsupported image format '{original_path.suffix}' for row {idx}."
            )
        record: Dict[str, object] = {
            "image": str(effective_path),
            "study_file": str(image_value),
            "row_index": int(idx),
            "resolved_image_path": str(effective_path),
            "original_image_path": str(original_path),
        }
        if id_column and id_column in df.columns:
            record[id_column] = row[id_column]
        records.append(record)
    return records


def _normalise_row_indices(value: Union[np.ndarray, torch.Tensor, Sequence[int]]) -> List[int]:
    """Convert batched row indices to a plain list of Python ``int``."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(int).tolist()
    if isinstance(value, np.ndarray):
        return value.astype(int).tolist()
    return [int(v) for v in value]


def _normalise_string_sequence(value: Union[Sequence[object], np.ndarray, str, None]) -> List[str]:
    """Normalise batched strings or paths returned by the DataLoader."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    return [str(value)]


def run_inference(
    input_path: Union[str, Path],
    artifacts: InferenceArtifacts,
    *,
    data_dir: Optional[Union[str, Path]] = None,
    out_csv: Optional[Union[str, Path]] = None,
    resume_csv: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Union[Dict[str, float], pd.DataFrame]:
    """Execute inference on a single volume or an entire CSV split.

    Parameters
    ----------
    input_path:
        Path to either a single ``nii.gz`` volume or to a CSV manifest.
    artifacts:
        Result of :func:`init_model`.
    data_dir:
        Root directory used to resolve relative paths from the CSV manifest.
        Ignored when ``input_path`` is a volume.
    out_csv:
        Destination CSV path used to persist predictions when processing a
        manifest. Required for CSV inputs.
    resume_csv:
        Optional CSV file with partial predictions. Samples whose
        ``study_file`` entries already appear in ``resume_csv`` are skipped.
        If ``resume_csv`` is omitted but ``out_csv`` already exists, the latter
        is used for resuming.
    batch_size, num_workers:
        DataLoader parameters for manifest inference.

    Returns
    -------
    dict or pandas.DataFrame
        A mapping of ``class -> probability`` for single volumes or a dataframe
        with predictions merged onto the manifest rows for CSV inputs.
    """

    input_path = Path(input_path)

    if input_path.is_file() and (_is_nifti_file(input_path) or _is_dicom_zip(input_path)):
        return predict_single_volume(input_path, artifacts)

    if not input_path.is_file() or input_path.suffix.lower() != ".csv":
        raise ValueError(
            "Input must be either a single nii.gz volume or a CSV manifest matching the training split format"
        )

    if out_csv is None:
        raise ValueError("'out_csv' must be provided when running inference over a CSV manifest")

    manifest_path = input_path
    out_csv = Path(out_csv)
    resume_path = Path(resume_csv) if resume_csv else None

    df = pd.read_csv(manifest_path)
    image_column = artifacts.image_column
    if image_column not in df.columns:
        raise ValueError(
            f"Manifest {manifest_path} does not contain the required image column '{image_column}'"
        )

    data_root: Optional[Path]
    if data_dir is not None:
        data_root = Path(data_dir)
    else:
        configured_root = artifacts.config.get("data_dir")
        data_root = Path(configured_root) if configured_root else manifest_path.parent

    resume_df: Optional[pd.DataFrame] = None
    candidate_paths: List[Path] = []
    if resume_path is not None and resume_path.exists():
        candidate_paths.append(resume_path)
    if out_csv.exists():
        candidate_paths.append(out_csv)
    if candidate_paths:
        latest_path = max(candidate_paths, key=lambda p: p.stat().st_mtime)
        resume_df = pd.read_csv(latest_path)
        LOGGER.info("Resuming predictions from %s", latest_path)

    processed: set[str] = set()
    results_df: pd.DataFrame
    if resume_df is not None and not resume_df.empty and image_column in resume_df.columns:
        results_df = resume_df.copy()
        processed = set(results_df[image_column].astype(str))
    else:
        results_df = pd.DataFrame()

    remaining_mask = ~df[image_column].astype(str).isin(processed)
    pending_indices = df.index[remaining_mask].tolist()

    if not pending_indices:
        LOGGER.info("No pending samples found in %s; writing existing results to %s", manifest_path, out_csv)
        if not results_df.empty:
            results_df.to_csv(out_csv, index=False)
        return results_df

    dicom_cache = DicomZipCache()
    try:
        records = _prepare_manifest_subset(
            df=df,
            image_column=image_column,
            pending_indices=pending_indices,
            data_root=data_root,
            id_column=artifacts.id_column,
            dicom_cache=dicom_cache,
        )
        dataset = Dataset(data=records, transform=artifacts.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=artifacts.device.type == "cuda",
            collate_fn=list_data_collate,
        )

        prob_columns = [f"prob_{cls}" for cls in artifacts.classes]
        base_columns = list(df.columns)
        desired_columns = base_columns + ["resolved_image_path", *prob_columns]

        if results_df.empty:
            results_df = pd.DataFrame(columns=desired_columns)
        else:
            for column in desired_columns:
                if column not in results_df.columns:
                    results_df[column] = np.nan

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(
                    device=artifacts.device,
                    dtype=artifacts.input_dtype,
                    non_blocking=True,
                )
                with _amp_context(artifacts):
                    logits = artifacts.model(images)
                    probs_tensor = _apply_probability_activation(
                        logits, artifacts.probability_activation
                    )
                probs = probs_tensor.float().cpu().numpy()

                study_files = list(batch.get("study_file", []))
                row_indices = _normalise_row_indices(batch.get("row_index", []))
                original_rows = df.loc[row_indices]
                original_records = original_rows.to_dict("records")
                resolved_paths = _normalise_string_sequence(batch.get("resolved_image_path"))
                if len(resolved_paths) != len(original_records):
                    resolved_paths = [
                        str(_resolve_volume_path(data_root, row_dict[image_column]))
                        for row_dict in original_records
                    ]

                batch_records: List[Dict[str, object]] = []
                for row_dict, prob_vec, study_file, resolved_path in zip(
                    original_records, probs, study_files, resolved_paths
                ):
                    result_row = dict(row_dict)
                    result_row["resolved_image_path"] = str(resolved_path)
                    for cls, value in zip(artifacts.classes, prob_vec):
                        result_row[f"prob_{cls}"] = float(value)
                    batch_records.append(result_row)
                    processed.add(str(study_file))

                batch_df = pd.DataFrame(batch_records)
                results_df = pd.concat([results_df, batch_df], ignore_index=True)
                results_df = results_df.drop_duplicates(subset=image_column, keep="last")
                results_df.to_csv(out_csv, index=False)

        order_map = {str(val): pos for pos, val in enumerate(df[image_column].astype(str))}
        results_df["__order"] = results_df[image_column].astype(str).map(order_map)
        results_df = results_df.sort_values("__order", kind="stable").drop(columns="__order")
        results_df.reset_index(drop=True, inplace=True)
        results_df.to_csv(out_csv, index=False)

        return results_df
    finally:
        dicom_cache.cleanup()


__all__ = [
    "InferenceArtifacts",
    "init_model",
    "predict_single_volume",
    "run_inference",
]

if __name__ == "__main__":
    artifacts = init_model()
    # run_inference(input_path="/mnt/data_nvme/workdir_4/ogk_hackathon/mosmed_hackathon/debug_val.csv", out_csv="/mnt/data_nvme/workdir_4/ogk_hackathon/pli_2/debug_val_pli_2.csv",
    #               artifacts=artifacts, num_workers=4, data_dir="")
    run_inference(input_path="/mnt/data_nvme/workdir_4/ogk_hackathon/mosmed_hackathon/unified_study_labels_fixed_kate.csv",
                  out_csv="/mnt/data_nvme/workdir_4/ogk_hackathon/pli_2/unified_study_labels_pli_2_e_15.csv",
                  artifacts=artifacts, data_dir="/astral-data/Minio/kate/data", num_workers=4)

    # single_path = "/astral-data/Minio/kate/data/raw/CT-RATE/dataset/valid_fixed/valid_1250/valid_1250_b/valid_1250_b_2.nii.gz"
    #
    # single_2 = "astral-data/Minio/kate/data/raw/clin_ev_01_02_03_05_2024/NIFTI/volumes/1.2.392.200036.9116.2.6.1.16.1613459359.1707454581.170345.nii.gz"
    #
    # dd = run_inference(input_path=single_path, artifacts=artifacts)
    # print(dd)
    # single_3 = "astral-data/Minio/kate/data/raw/clin_ev_01_02_03_05_2024/DICOM_zip/1.2.392.200036.9116.2.6.1.16.1613459359.1707454581.170345.zip"
    # dd2 = run_inference(input_path=single_path, artifacts=artifacts)
    # print(dd2)