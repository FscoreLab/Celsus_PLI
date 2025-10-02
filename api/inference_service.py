#!/usr/bin/env python3
"""
CT-CLIP Inference Service with LightGBM

Сервис для инференса:
1. Supervised модель (supervised_model.pt)
2. CT-CLIP модель (CT_VocabFine_v2.pt)
3. LightGBM модель для финального предсказания патологии
"""

import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import dicom2nifti
import joblib
import numpy as np
import pandas as pd
import pydicom
import shap
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Dict, Optional, Union
from mednext.inference import init_model, predict_single_volume

from CT_CLIP.ct_clip.ct_clip import CTCLIP
from ct_clip_classifier import SimpleCTCLIPClassifier
from ctclip_pathology_groups import CTCLIP_PATHOLOGY_GROUPS
from scripts.universal_ct_inference import UniversalCTInference
from transformer_maskgit.transformer_maskgit.ctvit import CTViT

logger = logging.getLogger(__name__)

CTCLIP_PATHOLOGIES = [
    "Pathological finding",
    "Pulmonary nodule",
    "Lymphangitic carcinomatosis",
    "Pneumonia",
    "Tuberculosis",
    "Bronchiectasis",
    "Chronic bronchitis",
    "Bronchial obstruction",
    "Pulmonary emphysema",
    "Pulmonary fibrosis",
    "Pneumoconiosis",
    "Pulmonary edema",
    "Atelectasis",
    "Pleural effusion",
    "Pleural thickening",
    "Pleural plaques",
    "Pneumothorax",
    "Hemothorax",
    "Consolidation",
    "Ground-glass opacity",
    "Crazy-paving pattern",
    "Tree-in-bud pattern",
    "Mosaic attenuation",
    "Cavitation",
    "Pulmonary embolism",
    "Chronic thromboembolic disease",
    "Main pulmonary artery enlargement",
    "Pulmonary arteriovenous malformation",
    "Mediastinal or hilar lymphadenopathy",
    "Sarcoidosis",
    "Thymoma",
    "Lymphoma",
    "Mediastinal cyst",
    "Retrosternal goiter",
    "Pericardial effusion",
    "Cardiac chamber enlargement",
    "Coronary artery calcification",
    "Aortic dissection",
    "Rib fracture",
    "Vertebral compression fracture",
    "Bone metastases",
    "Chest wall mass or invasion",
    "Subcutaneous emphysema",
    "Diaphragmatic hernia",
    "Hiatal hernia",
    "Pneumomediastinum",
    "Hydropneumothorax",
    "Radiation pneumonitis",
    "Postoperative changes (lobectomy or pneumonectomy)",
    "Scoliosis",
    "Pectus excavatum or carinatum",
    "Degenerative spine changes",
    "Thyroid nodule",
    "Breast mass",
    "Other pathology",
]

MOSMED_PATHOLOGIES = [
    "ribs_fracture",
    "pleural_effusion",
    "aorta_pathology",
    "pulmonary_trunk_pathology",
    "cancer",
    "atelectasis",
    "covid",
    "infiltrate",
    "emphysema",
    "paracardial_fat_pathology",
    "fibrosis",
    "airiness_decrease",
    "pneumothorax",
    "coronary_calcium",
    "osteo_fracture",
]


class DICOMValidator:
    """Валидация и обработка поврежденных/неполных DICOM файлов."""

    MIN_SLICES = 20

    REQUIRED_TAGS = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Modality",
    ]

    IMAGE_TAGS = [
        "Rows",
        "Columns",
        "PixelData",
    ]

    @staticmethod
    def validate_dicom_file(file_path: str, check_pixels: bool = True) -> Dict:
        """
        Валидирует DICOM файл и возвращает отчет.

        Parameters
        ----------
        file_path : str
            Путь к DICOM файлу
        check_pixels : bool
            Проверять ли наличие и читаемость pixel data

        Returns
        -------
        dict
            {
                'valid': bool,
                'can_process': bool,
                'errors': List[str],
                'warnings': List[str],
                'missing_tags': List[str],
                'metadata': Dict or None
            }
        """
        result = {
            "valid": False,
            "can_process": False,
            "errors": [],
            "warnings": [],
            "missing_tags": [],
            "metadata": None,
        }

        try:
            if check_pixels:
                dcm = pydicom.dcmread(file_path, force=True)
            else:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
        except Exception as e:
            result["errors"].append(f"Невозможно прочитать файл: {str(e)}")
            return result

        missing_required = []
        for tag in DICOMValidator.REQUIRED_TAGS:
            if not hasattr(dcm, tag):
                missing_required.append(tag)

        if missing_required:
            result["errors"].append(f"Отсутствуют обязательные теги: {', '.join(missing_required)}")
            result["missing_tags"] = missing_required
            return result

        if hasattr(dcm, "Modality"):
            modality = str(dcm.Modality).upper()
            if modality != "CT":
                result["warnings"].append(f"Модальность {modality}, ожидалась CT")

        if check_pixels:
            missing_image = []
            for tag in DICOMValidator.IMAGE_TAGS:
                if not hasattr(dcm, tag):
                    missing_image.append(tag)

            if missing_image:
                result["warnings"].append(f"Отсутствуют теги изображения: {', '.join(missing_image)}")
            else:
                try:
                    pixel_array = dcm.pixel_array
                    if pixel_array is None or pixel_array.size == 0:
                        result["errors"].append("PixelData пуст")
                        return result

                    expected_shape = (int(dcm.Rows), int(dcm.Columns))
                    if pixel_array.shape[:2] != expected_shape:
                        result["warnings"].append(
                            f"Размер изображения {pixel_array.shape[:2]} не соответствует тегам {expected_shape}"
                        )
                except Exception as e:
                    result["errors"].append(f"Ошибка чтения pixel data: {str(e)}")
                    return result

        result["metadata"] = {
            "study_uid": str(dcm.StudyInstanceUID),
            "series_uid": str(dcm.SeriesInstanceUID),
            "sop_uid": str(dcm.SOPInstanceUID),
            "modality": str(dcm.Modality) if hasattr(dcm, "Modality") else "Unknown",
            "instance_number": int(dcm.InstanceNumber) if hasattr(dcm, "InstanceNumber") else None,
            "slice_location": float(dcm.SliceLocation) if hasattr(dcm, "SliceLocation") else None,
        }

        result["valid"] = len(result["errors"]) == 0
        result["can_process"] = result["valid"]

        return result

    @staticmethod
    def validate_series(dicom_paths: List[str], check_pixels: bool = False) -> Dict:
        """
        Валидирует серию DICOM файлов.

        Parameters
        ----------
        dicom_paths : List[str]
            Список путей к DICOM файлам серии
        check_pixels : bool
            Проверять ли pixel data (медленно)

        Returns
        -------
        dict
            {
                'valid': bool,
                'can_process': bool,
                'num_files': int,
                'num_valid': int,
                'num_corrupted': int,
                'errors': List[str],
                'warnings': List[str],
                'valid_files': List[str],
                'corrupted_files': List[Dict]
            }
        """
        result = {
            "valid": False,
            "can_process": False,
            "num_files": len(dicom_paths),
            "num_valid": 0,
            "num_corrupted": 0,
            "errors": [],
            "warnings": [],
            "valid_files": [],
            "corrupted_files": [],
        }

        if not dicom_paths:
            result["errors"].append("Пустой список DICOM файлов")
            return result

        for file_path in dicom_paths:
            validation = DICOMValidator.validate_dicom_file(file_path, check_pixels=check_pixels)

            if validation["can_process"]:
                result["num_valid"] += 1
                result["valid_files"].append(file_path)
            else:
                result["num_corrupted"] += 1
                result["corrupted_files"].append(
                    {"file": file_path, "errors": validation["errors"], "warnings": validation["warnings"]}
                )

        if result["num_valid"] < DICOMValidator.MIN_SLICES:
            result["errors"].append(
                f"Недостаточно валидных срезов: {result['num_valid']}, требуется минимум {DICOMValidator.MIN_SLICES}"
            )
            result["can_process"] = False
        else:
            result["can_process"] = True
            result["valid"] = True

        if result["num_corrupted"] > 0:
            result["warnings"].append(f"Пропущено {result['num_corrupted']} поврежденных файлов")

        return result

    @staticmethod
    def filter_valid_files(dicom_paths: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        Фильтрует файлы, оставляя только валидные.

        Returns
        -------
        Tuple[List[str], List[Dict]]
            (valid_files, rejected_files_info)
        """
        valid_files = []
        rejected_files = []

        for file_path in dicom_paths:
            validation = DICOMValidator.validate_dicom_file(file_path, check_pixels=False)

            if validation["can_process"]:
                valid_files.append(file_path)
            else:
                rejected_files.append(
                    {
                        "file": os.path.basename(file_path),
                        "errors": validation["errors"],
                        "warnings": validation["warnings"],
                    }
                )

        return valid_files, rejected_files


class DICOMSeriesSelector:
    """Выбирает подходящую DICOM серию из архива."""

    @staticmethod
    def extract_series_from_zip(zip_path: str, extract_dir: str) -> Dict[str, List[str]]:
        """
        Извлекает ZIP и группирует DICOM файлы по сериям с валидацией.

        Returns:
            Dict[series_uid, list_of_dicom_paths]
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Находим все DICOM файлы с валидацией
        dicom_files = []
        total_files = 0
        corrupted_count = 0

        for root, _, files in os.walk(extract_dir):
            for file in files:
                # Пропускаем системные файлы
                if file.startswith(".") or file.endswith((".txt", ".pdf", ".jpg", ".png")):
                    continue

                file_path = os.path.join(root, file)
                total_files += 1

                # Используем валидатор для проверки файла
                validation = DICOMValidator.validate_dicom_file(file_path, check_pixels=False)

                if validation["can_process"]:
                    dicom_files.append(file_path)
                else:
                    corrupted_count += 1
                    logger.debug(f"Пропущен файл {os.path.basename(file_path)}: {validation['errors']}")

        logger.info(f"Обработано {total_files} файлов: {len(dicom_files)} валидных, {corrupted_count} пропущено")

        # Группируем по SeriesInstanceUID
        series_dict = {}
        for dcm_path in dicom_files:
            try:
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
                series_uid = str(dcm.SeriesInstanceUID)
                if series_uid not in series_dict:
                    series_dict[series_uid] = []
                series_dict[series_uid].append(dcm_path)
            except Exception as e:
                logger.warning(f"Ошибка чтения DICOM {dcm_path}: {e}")
                continue

        return series_dict

    @staticmethod
    def select_best_series(series_dict: Dict[str, List[str]]) -> Tuple[str, List[str]]:
        """
        Выбирает лучшую серию (стандартную или лёгочную) с валидацией.

        Приоритет:
        1. Лёгочная серия (LUNG kernel) с >= 20 срезов
        2. Стандартная серия (STANDARD kernel) с >= 20 срезов
        3. Любая серия с наибольшим количеством срезов (>= 20)

        Returns:
            (series_uid, list_of_dicom_paths) - только валидные файлы
        """
        if not series_dict:
            raise ValueError("Нет DICOM серий в архиве")

        lung_series = []
        standard_series = []
        all_series = []

        for series_uid, dicom_paths in series_dict.items():
            if not dicom_paths:
                continue

            # Валидируем серию
            validation = DICOMValidator.validate_series(dicom_paths, check_pixels=False)

            # Пропускаем серии с недостаточным количеством валидных срезов
            if not validation["can_process"]:
                logger.info(
                    f"Серия {series_uid} пропущена: {validation['num_valid']} валидных срезов "
                    f"из {validation['num_files']} (требуется минимум {DICOMValidator.MIN_SLICES})"
                )
                if validation["num_corrupted"] > 0:
                    logger.debug(f"Поврежденных файлов: {validation['num_corrupted']}")
                continue

            # Используем только валидные файлы
            valid_paths = validation["valid_files"]

            # Читаем первый валидный файл для анализа
            try:
                dcm = pydicom.dcmread(valid_paths[0], stop_before_pixels=True, force=True)

                # Анализируем kernel/reconstruction
                kernel = ""
                if hasattr(dcm, "ConvolutionKernel"):
                    kernel = str(dcm.ConvolutionKernel).upper()
                elif hasattr(dcm, "FilterType"):
                    kernel = str(dcm.FilterType).upper()

                series_info = {
                    "series_uid": series_uid,
                    "paths": valid_paths,  # Используем только валидные файлы
                    "num_slices": len(valid_paths),
                    "kernel": kernel,
                    "validation": validation,
                }

                all_series.append(series_info)

                # Классифицируем серию
                if any(k in kernel for k in ["LUNG", "B70", "B80", "FC51", "FC81"]):
                    lung_series.append(series_info)
                elif any(k in kernel for k in ["STANDARD", "B30", "B31", "FC30"]):
                    standard_series.append(series_info)

            except Exception as e:
                logger.warning(f"Ошибка анализа серии {series_uid}: {e}")
                continue

        # Проверяем, что есть хотя бы одна подходящая серия
        if not all_series:
            raise ValueError(
                f"Не найдено серий с минимальным количеством валидных срезов ({DICOMValidator.MIN_SLICES})"
            )

        # Выбираем лучшую серию
        if lung_series:
            # Выбираем лёгочную серию с наибольшим количеством срезов
            best = max(lung_series, key=lambda x: x["num_slices"])
            logger.info(
                f"Выбрана лёгочная серия: {best['series_uid']} "
                f"({best['num_slices']} валидных срезов, kernel: {best['kernel']})"
            )
            return best["series_uid"], best["paths"]
        elif standard_series:
            # Выбираем стандартную серию с наибольшим количеством срезов
            best = max(standard_series, key=lambda x: x["num_slices"])
            logger.info(
                f"Выбрана стандартная серия: {best['series_uid']} "
                f"({best['num_slices']} валидных срезов, kernel: {best['kernel']})"
            )
            return best["series_uid"], best["paths"]
        elif all_series:
            # Выбираем любую серию с наибольшим количеством срезов
            best = max(all_series, key=lambda x: x["num_slices"])
            logger.info(
                f"Выбрана серия по умолчанию: {best['series_uid']} "
                f"({best['num_slices']} валидных срезов, kernel: {best['kernel']})"
            )
            return best["series_uid"], best["paths"]
        else:
            raise ValueError("Не удалось найти подходящую DICOM серию")


class SupervisedModelInference:
    """Инференс Supervised модели."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.pathologies = None

    def load_model(self):
        """Загружает supervised модель."""
        logger.info("Загружаем supervised модель...")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if "target_pathologies" in checkpoint:
            self.pathologies = checkpoint["target_pathologies"]
            logger.info(f"Загружено {len(self.pathologies)} патологий из checkpoint'а")
        else:
            self.pathologies = MOSMED_PATHOLOGIES
            logger.warning("target_pathologies не найдено в checkpoint'е, используем дефолтный список")

        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(tokenizer))

        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )

        ct_clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
        )

        self.model = SimpleCTCLIPClassifier(
            ct_clip,
            num_classes=len(self.pathologies),
            latent_norm="layernorm",  # для новой модели используется layernorm
            reinit_head=False,
        )
        self.model.tokenizer = tokenizer

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()
        logger.info("Supervised модель загружена успешно!")

    def predict(self, nifti_path: str) -> Dict[str, float]:
        """Делает предсказания для NIFTI файла."""
        if self.model is None:
            raise RuntimeError("Модель не загружена!")

        ct_inference = UniversalCTInference(model=None, device=self.device, verbose=False)
        tensor = ct_inference.nii_to_tensor(nifti_path)
        tensor = tensor.unsqueeze(1)  # [1, 1, 240, 480, 480]

        with torch.inference_mode():
            tensor = tensor.to(self.device)
            outputs = self.model(tensor)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = {}
        for i, pathology in enumerate(self.pathologies):
            predictions[f"supervised_{pathology}"] = float(probs[i])

        return predictions

    def unload(self):
        """Выгружает модель из памяти."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()


class CTCLIPInferenceService:
    """Сервис для инференса обеих моделей."""

    def __init__(
        self,
        supervised_model_path: str,
        ctclip_model_path: str,
        device: str = "cuda",
    ):
        self.supervised_model_path = supervised_model_path
        self.ctclip_model_path = ctclip_model_path
        self.device = device

        self.supervised_inference = SupervisedModelInference(supervised_model_path, device)
        self.ctclip_inference = UniversalCTInference(model_path=ctclip_model_path, device=device, verbose=False)

    def process_zip_archive(self, zip_path: str) -> Dict:
        """
        Обрабатывает ZIP архив с DICOM файлами.

        Returns:
            Dict с предсказаниями обеих моделей
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = os.path.join(temp_dir, "dicom_extract")
            nifti_dir = os.path.join(temp_dir, "nifti")
            os.makedirs(extract_dir, exist_ok=True)
            os.makedirs(nifti_dir, exist_ok=True)

            # Извлекаем и выбираем лучшую серию
            logger.info("Извлекаем DICOM файлы...")
            series_dict = DICOMSeriesSelector.extract_series_from_zip(zip_path, extract_dir)
            logger.info(f"Найдено {len(series_dict)} серий")

            series_uid, dicom_paths = DICOMSeriesSelector.select_best_series(series_dict)
            logger.info(f"Выбрана серия: {series_uid}")

            # Получаем Study UID из первого DICOM файла
            dcm = pydicom.dcmread(dicom_paths[0], stop_before_pixels=True)
            study_uid = str(dcm.StudyInstanceUID)

            # Конвертируем выбранную серию в NIFTI
            logger.info("Конвертируем DICOM в NIFTI...")
            series_dir = os.path.join(extract_dir, "selected_series")
            os.makedirs(series_dir, exist_ok=True)

            # Копируем только выбранные файлы
            import shutil

            for dcm_path in dicom_paths:
                shutil.copy(dcm_path, series_dir)

            nifti_file = os.path.join(nifti_dir, f"{series_uid}.nii.gz")

            try:
                dicom2nifti.dicom_series_to_nifti(series_dir, nifti_file, reorient_nifti=False)
            except Exception as e:
                logger.error(f"Ошибка конвертации DICOM в NIFTI: {e}")
                raise

            # Запускаем инференс обеих моделей последовательно
            predictions = {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "probabilities": {},
            }

            # 1. Supervised модель
            logger.info("Загружаем supervised модель...")
            self.supervised_inference.load_model()
            supervised_preds = self.supervised_inference.predict(nifti_file)
            predictions["probabilities"].update(supervised_preds)
            self.supervised_inference.unload()
            logger.info("Supervised модель выгружена")

            # 2. CT-CLIP модель
            logger.info("Загружаем CT-CLIP модель...")
            ctclip_result = self.ctclip_inference.infer(nifti_file, custom_pathologies=CTCLIP_PATHOLOGIES)

            for pathology, prob in ctclip_result["pathology_predictions"].items():
                predictions["probabilities"][f"ctclip_{pathology}"] = prob

            # Выгружаем CT-CLIP модель
            if hasattr(self.ctclip_inference, "model") and self.ctclip_inference.model is not None:
                del self.ctclip_inference.model
                self.ctclip_inference.model = None
                torch.cuda.empty_cache()
            logger.info("CT-CLIP модель выгружена")

            return predictions


class DiffusionClassifierInference:
    """Класс для инференса diffusion classifier на CT срезах."""

    def __init__(self, model_path: str, device: str = "cuda", image_size: int = 256):
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.model = None

    def load_model(self):
        """Загружает diffusion classifier."""
        logger.info("Загружаем diffusion classifier...")

        # Импортируем diffusion модули
        diffusion_path = str(Path(__file__).parent.parent / "diffusion_anomaly")
        if diffusion_path not in sys.path:
            sys.path.insert(0, diffusion_path)

        from diffusion_anomaly.dist_util import load_state_dict
        from diffusion_anomaly.script_util import create_classifier

        # Создаем classifier
        self.model = create_classifier(
            image_size=self.image_size,
            classifier_use_fp16=False,
            classifier_width=32,
            classifier_depth=2,
            classifier_attention_resolutions="32,16,8",
            classifier_use_scale_shift_norm=True,
            classifier_resblock_updown=True,
            classifier_pool="attention",
            dataset="ct-rate",
        )

        # Загружаем веса
        logger.info(f"Loading diffusion classifier from: {self.model_path}")
        state_dict = load_state_dict(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        logger.info("Diffusion classifier загружен успешно!")

    def _prepare_slices_from_nifti(self, nifti_path: str, num_slices: int = None) -> torch.Tensor:
        """Подготавливает срезы из NIFTI файла с center crop + resize."""
        ct_inference = UniversalCTInference(model=None, device=self.device, verbose=False)
        volume_tensor = ct_inference.nii_to_tensor(nifti_path)

        if volume_tensor.dim() == 4 and volume_tensor.shape[0] == 1:
            volume_tensor = volume_tensor[0]

        # Срезы из середины (5-95%)
        depth = volume_tensor.shape[0]
        start_idx = int(depth * 0.05)
        end_idx = int(depth * 0.95)
        mid_slices = list(range(start_idx, end_idx))

        if num_slices and len(mid_slices) > num_slices:
            indices = np.linspace(0, len(mid_slices) - 1, num_slices, dtype=int)
            mid_slices = [mid_slices[i] for i in indices]

        slices = []
        for idx in mid_slices:
            slice_2d = volume_tensor[idx]
            slice_tensor = slice_2d.unsqueeze(0)

            # Center crop + resize
            if slice_tensor.shape[1:] != (self.image_size, self.image_size):
                h, w = slice_tensor.shape[1], slice_tensor.shape[2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                slice_tensor = slice_tensor[:, top : top + min_dim, left : left + min_dim]
                if min_dim != self.image_size:
                    slice_tensor = F.interpolate(
                        slice_tensor.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

            slices.append(slice_tensor)

        return torch.stack(slices)

    def predict(self, nifti_path: str, batch_size: int = 16) -> Dict[str, float]:
        """Делает предсказания для NIFTI файла."""
        if self.model is None:
            raise RuntimeError("Модель не загружена!")

        slices = self._prepare_slices_from_nifti(nifti_path, num_slices=None)

        all_probs = []
        with torch.inference_mode():
            for i in range(0, len(slices), batch_size):
                batch = slices[i : i + batch_size].to(self.device)
                t = torch.zeros(batch.shape[0], dtype=torch.long, device=self.device)
                logits = self.model(batch, t)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

        all_probs = np.array(all_probs)
        predictions = {
            "diffusion_classifier_mean_probability": float(all_probs.mean()),
            "diffusion_classifier_max_probability": float(all_probs.max()),
            "diffusion_classifier_std_probability": float(all_probs.std()),
        }

        return predictions


class DiffusionReconstructionInference:
    """Класс для инференса diffusion модели с guided reconstruction."""

    def __init__(
        self,
        diffusion_model_path: str,
        classifier_model: "DiffusionClassifierInference",
        device: str = "cuda",
        image_size: int = 256,
        classifier_scale: float = 200.0,
        noise_level: int = 500,
        num_inference_steps: int = 20,
    ):
        self.diffusion_model_path = diffusion_model_path
        self.classifier_model = classifier_model  # Используем уже загруженный classifier
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.classifier_scale = classifier_scale
        self.noise_level = noise_level
        self.num_inference_steps = num_inference_steps
        self.diffusion_model = None
        self.diffusion = None

    def load_model(self):
        """Загружает diffusion UNet + diffusion process."""
        logger.info("Загружаем diffusion reconstruction модель...")

        diffusion_path = str(Path(__file__).parent.parent / "diffusion_anomaly")
        if diffusion_path not in sys.path:
            sys.path.insert(0, diffusion_path)

        from diffusion_anomaly.gaussian_diffusion import (
            GaussianDiffusion,
            LossType,
            ModelMeanType,
            ModelVarType,
            get_named_beta_schedule,
        )
        from diffusion_anomaly.unet import UNetModel

        # Создаем UNet модель
        self.diffusion_model = UNetModel(
            image_size=self.image_size,
            in_channels=1,
            model_channels=128,
            out_channels=2,
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0.0,
            channel_mult=(1, 2, 4, 8),
            num_classes=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
        )

        # Загружаем веса
        checkpoint = torch.load(self.diffusion_model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.diffusion_model.load_state_dict(checkpoint)

        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
        logger.info("Diffusion UNet загружен")

        # Создаем Gaussian Diffusion process
        diffusion_steps = 1000
        betas = get_named_beta_schedule("linear", diffusion_steps)

        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.LEARNED_RANGE,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        logger.info("Diffusion reconstruction загружена успешно!")

    def _prepare_slices_from_nifti(self, nifti_path: str, num_slices: int = 16) -> torch.Tensor:
        """Подготавливает равномерно распределенные срезы с center crop + resize."""
        ct_inference = UniversalCTInference(model=None, device=self.device, verbose=False)
        volume_tensor = ct_inference.nii_to_tensor(nifti_path)

        if volume_tensor.dim() == 4 and volume_tensor.shape[0] == 1:
            volume_tensor = volume_tensor[0]

        depth = volume_tensor.shape[0]
        start_idx = int(depth * 0.05)
        end_idx = int(depth * 0.95)
        available_slices = end_idx - start_idx

        if available_slices > num_slices:
            indices = np.linspace(0, available_slices - 1, num_slices, dtype=int)
            mid_slices = [start_idx + i for i in indices]
        else:
            mid_slices = list(range(start_idx, end_idx))

        slices = []
        for idx in mid_slices:
            slice_2d = volume_tensor[idx]
            slice_tensor = slice_2d.unsqueeze(0)

            # Center crop + resize
            if slice_tensor.shape[1:] != (self.image_size, self.image_size):
                h, w = slice_tensor.shape[1], slice_tensor.shape[2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                slice_tensor = slice_tensor[:, top : top + min_dim, left : left + min_dim]
                if min_dim != self.image_size:
                    slice_tensor = F.interpolate(
                        slice_tensor.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )[0]

            slices.append(slice_tensor)

        return torch.stack(slices)

    def _create_cond_fn(self):
        """Создает classifier guidance function."""

        def cond_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = self.classifier_model.model(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                grad = torch.autograd.grad(selected.sum(), x_in)[0]
                return grad, grad * self.classifier_scale

        return cond_fn

    def predict(self, nifti_path: str, batch_size: int = 8) -> Dict[str, float]:
        """Делает guided reconstruction и возвращает anomaly scores."""
        if self.diffusion_model is None:
            raise RuntimeError("Diffusion модель не загружена!")

        slices = self._prepare_slices_from_nifti(nifti_path, num_slices=16)
        cond_fn = self._create_cond_fn()
        all_errors = []

        for i in range(0, len(slices), batch_size):
            batch = slices[i : i + batch_size].to(self.device)
            bsz = batch.shape[0]

            target_y = torch.zeros(bsz, dtype=torch.long, device=self.device)
            model_kwargs = {"y": target_y}

            with torch.no_grad():  # no_grad для diffusion из-за enable_grad в cond_fn
                # Forward encoding
                timesteps_forward = torch.linspace(
                    0, self.noise_level - 1, self.num_inference_steps, dtype=torch.long, device=self.device
                )
                current = batch

                for j in range(len(timesteps_forward) - 1):
                    t_batch = timesteps_forward[j].repeat(bsz).to(self.device)
                    out = self.diffusion.ddim_reverse_sample(
                        self.diffusion_model, current, t_batch, clip_denoised=True, model_kwargs=model_kwargs, eta=0.0
                    )
                    current = out["sample"]

                x_encoded = current

                # Backward decoding with guidance
                timesteps_backward = torch.linspace(
                    self.noise_level - 1, 0, self.num_inference_steps, dtype=torch.long, device=self.device
                )
                current = x_encoded

                for j in range(len(timesteps_backward) - 1):
                    t_batch = timesteps_backward[j].repeat(bsz).to(self.device)
                    out = self.diffusion.ddim_sample(
                        self.diffusion_model,
                        current,
                        t_batch,
                        clip_denoised=True,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                        eta=0.0,
                    )
                    current = out["sample"]

                reconstructed = current

                # L1 anomaly metric
                diff = (batch - reconstructed).abs()
                anomaly_map = diff.sum(dim=1, keepdim=True)
                errors = anomaly_map.mean(dim=[1, 2, 3]).cpu().numpy()
                errors = np.nan_to_num(errors, nan=0.0, posinf=1e6, neginf=-1e6)
                all_errors.extend(errors)

        all_errors = np.array(all_errors)
        predictions = {
            "diffusion_reconstruction_mean": float(all_errors.mean()),
            "diffusion_reconstruction_max": float(all_errors.max()),
            "diffusion_reconstruction_std": float(all_errors.std()),
        }

        return predictions


class LightGBMInference:
    """Инференс LightGBM модели с SHAP анализом."""

    def __init__(self, model_path: str, optimal_threshold: float = 0.5):
        self.model_path = model_path
        self.optimal_threshold = optimal_threshold
        self.model = None
        self.feature_names = None
        self.shap_explainer = None

        # Списки патологий для маппинга фичей
        self.supervised_pathologies = MOSMED_PATHOLOGIES
        self.ctclip_pathologies = CTCLIP_PATHOLOGIES


    @staticmethod
    def _ensure_prefixed(d: Dict[str, float], prefix: str) -> Dict[str, float]:
        """Возвращает dict с гарантированным префиксом у ключей."""
        if not d:
            return {}
        prefixed = {}
        for k, v in d.items():
            if k.startswith(prefix):
                prefixed[k] = float(v)
            else:
                prefixed[f"{prefix}{k}"] = float(v)
        return prefixed


    def load_model(self):
        """Загружает LightGBM модель."""
        logger.info(f"Загружаем LightGBM модель из {self.model_path}...")

        # Загружаем модель (это может быть Pipeline или просто модель)
        self.model = joblib.load(self.model_path)

        # Получаем список фичей
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        elif hasattr(self.model, "named_steps"):
            # Если это Pipeline, получаем фичи из последнего шага
            final_step = list(self.model.named_steps.values())[-1]
            if hasattr(final_step, "feature_name_"):
                self.feature_names = list(final_step.feature_name_)

        num_features = len(self.feature_names) if self.feature_names else "неизвестно"
        logger.info(f"LightGBM модель загружена, ожидается {num_features} фичей")

    def prepare_features(
        self,
        supervised_probs: Dict[str, float],
        ctclip_probs: Dict[str, float],
        diffusion_classifier_probs: Dict[str, float] = None,
        diffusion_reconstruction_scores: Dict[str, float] = None,
        mednext_preds: Optional[Dict[str, float]] = None,
        fvlm_preds: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Подготавливает фичи для LightGBM из предсказаний всех моделей.

        Parameters
        ----------
        supervised_probs : dict
            Предсказания supervised модели {pathology: probability}
        ctclip_probs : dict
            Предсказания CT-CLIP модели {pathology: probability}
        diffusion_classifier_probs : dict, optional
            Предсказания diffusion classifier (mean, max, std)
        diffusion_reconstruction_scores : dict, optional
            Anomaly scores от diffusion reconstruction (mean, max, std)
        mednext_preds: dict (str, float)
            classification scores for 22 classes including "normal" (no pathology)

        Returns
        -------
        pd.DataFrame
            Датафрейм с одной строкой и всеми необходимыми фичами
        """
        features = {}

        # Добавляем supervised фичи
        for pathology in self.supervised_pathologies:
            feature_name = f"supervised_{pathology}"
            features[feature_name] = supervised_probs.get(feature_name, 0.0)

        # Добавляем CT-CLIP фичи
        for pathology in self.ctclip_pathologies:
            feature_name = f"ctclip_{pathology}"
            features[feature_name] = ctclip_probs.get(feature_name, 0.0)

        if mednext_preds:
            mednext_norm = self._ensure_prefixed(mednext_preds, "kolyan_")
            for k, v in mednext_norm.items():
                features[k] = float(v)

        if fvlm_preds:
            fvlm_norm = self._ensure_prefixed(fvlm_preds, "okhr_")
            for k, v in fvlm_norm.items():
                features[k] = float(v)

        def _collect(prefix: str) -> List[float]:
            return [v for k, v in features.items() if k.startswith(prefix)]

        # Вычисляем агрегированные фичи
        supervised_values = [features[f"supervised_{p}"] for p in self.supervised_pathologies]
        ctclip_values = [features[f"ctclip_{p}"] for p in self.ctclip_pathologies]
        mednext_vals = _collect("kolyan_") # TODO integrate further
        fvlm_vals = _collect("okhr_")  # TODO integrate further

        features["supervised_mean_probability"] = np.mean(supervised_values)
        features["ctclip_mean_probability"] = np.mean(ctclip_values)
        features["supervised_max_probability"] = np.max(supervised_values)
        features["ctclip_max_probability"] = np.max(ctclip_values)
        features["supervised_top3_mean"] = np.mean(sorted(supervised_values, reverse=True)[:3])
        features["ctclip_top3_mean"] = np.mean(sorted(ctclip_values, reverse=True)[:3])

        # Добавляем групповые фичи для CT-CLIP
        for group_name, pathologies in CTCLIP_PATHOLOGY_GROUPS.items():
            # Собираем вероятности для патологий в группе
            group_values = [features[f"ctclip_{pathology}"] for pathology in pathologies 
                            if f"ctclip_{pathology}" in features]
            
            if group_values:
                # Max - максимальная вероятность в группе
                features[f"ctclip_group_{group_name}_max"] = np.max(group_values)
                
                # Mean - средняя вероятность в группе
                features[f"ctclip_group_{group_name}_mean"] = np.mean(group_values)
                
                # Count high - количество патологий с p > 0.5
                features[f"ctclip_group_{group_name}_count_high"] = np.sum(np.array(group_values) > 0.5)
                
                # Std - стандартное отклонение (для групп с >1 патологией)
                if len(group_values) > 1:
                    features[f"ctclip_group_{group_name}_std"] = np.std(group_values)
                else:
                    features[f"ctclip_group_{group_name}_std"] = 0.0

        # Добавляем diffusion classifier фичи если есть
        if diffusion_classifier_probs:
            features.update(diffusion_classifier_probs)

        # Добавляем diffusion reconstruction фичи если есть
        if diffusion_reconstruction_scores:
            features.update(diffusion_reconstruction_scores)

        # Создаем датафрейм
        df = pd.DataFrame([features])

        # Фильтруем только нужные фичи, если список известен
        if self.feature_names:
            # Добавляем недостающие фичи как 0
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0.0
            # Оставляем только нужные фичи в правильном порядке
            df = df[self.feature_names]

        return df

    def predict(self, features_df: pd.DataFrame) -> Dict:
        """
        Делает предсказание и SHAP анализ.

        Parameters
        ----------
        features_df : pd.DataFrame
            Датафрейм с фичами

        Returns
        -------
        dict
            Результаты предсказания:
            - probability: вероятность патологии
            - prediction: бинарное предсказание (0/1)
            - most_dangerous_pathology: название патологии с максимальным SHAP вкладом
            - shap_values: dict с SHAP значениями для топ фичей
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена!")

        # Предсказание вероятности
        prob = self.model.predict_proba(features_df)[0, 1]
        prediction = 1 if prob >= self.optimal_threshold else 0

        # SHAP анализ для определения most dangerous pathology
        try:
            # Создаем explainer если еще не создан
            if self.shap_explainer is None:
                logger.info("Создаем SHAP explainer...")
                # Получаем базовую модель из pipeline если нужно
                model_to_explain = self.model
                if hasattr(self.model, "named_steps"):
                    model_to_explain = self.model.named_steps["lr"]  # последний шаг

                self.shap_explainer = shap.TreeExplainer(model_to_explain)

            # Вычисляем SHAP values
            shap_values = self.shap_explainer.shap_values(features_df)

            # Если бинарная классификация, берем values для класса 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Находим фичу с максимальным положительным SHAP вкладом
            shap_values_flat = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Фильтруем только положительные SHAP значения (увеличивающие вероятность патологии)
            positive_mask = shap_values_flat > 0
            max_shap_idx = None  # Инициализируем для дальнейшего использования
            
            if not np.any(positive_mask):
                # Нет положительных SHAP вкладов - патология не обнаружена
                most_dangerous_pathology = "No specific pathology detected"
                most_dangerous_feature = None
            else:
                # Находим максимальный положительный SHAP вклад среди КОНКРЕТНЫХ патологий
                # Исключаем агрегированные фичи (mean, max, top3, std) и групповые фичи (_group_)
                feature_names = features_df.columns
                pathology_mask = positive_mask & ~np.array([
                    any(x in str(feat) for x in ['_mean', '_max', '_top3', '_std', '_group_'])
                    for feat in feature_names
                ])
                
                if np.any(pathology_mask):
                    # Берем максимальный SHAP среди конкретных патологий
                    pathology_shap_values = shap_values_flat.copy()
                    pathology_shap_values[~pathology_mask] = -np.inf
                    max_shap_idx = np.argmax(pathology_shap_values)
                else:
                    # Если нет конкретных патологий, берем максимальный положительный
                    positive_shap_values = shap_values_flat.copy()
                    positive_shap_values[~positive_mask] = -np.inf
                    max_shap_idx = np.argmax(positive_shap_values)
                
                most_dangerous_feature = features_df.columns[max_shap_idx]
                most_dangerous_pathology = self._feature_to_pathology_name(most_dangerous_feature)

            # Собираем топ-5 положительных SHAP вкладов
            top_shap_indices = np.argsort(shap_values_flat)[-5:][::-1]
            top_shap_features = {}
            for idx in top_shap_indices:
                if shap_values_flat[idx] > 0:  # Только положительные вклады
                    feat_name = features_df.columns[idx]
                    top_shap_features[feat_name] = float(shap_values_flat[idx])

            if most_dangerous_feature is not None:
                shap_value = shap_values_flat[max_shap_idx]
                logger.info(f"Most dangerous pathology: {most_dangerous_pathology} (SHAP: {shap_value:.4f})")
            else:
                logger.info(f"Most dangerous pathology: {most_dangerous_pathology}")

        except Exception as e:
            logger.warning(f"Ошибка SHAP анализа: {e}")
            most_dangerous_pathology = "Unknown"
            top_shap_features = {}

        return {
            "probability": float(prob),
            "prediction": int(prediction),
            "most_dangerous_pathology": most_dangerous_pathology,
            "top_shap_features": top_shap_features,
        }

    def _feature_to_pathology_name(self, feature_name: str) -> str:
        """
        Преобразует название фичи в человеко-читаемое название патологии.

        Parameters
        ----------
        feature_name : str
            Название фичи (например, "supervised_cancer" или "ctclip_Pneumonia")

        Returns
        -------
        str
            Название патологии
        """
        # Удаляем префикс
        if feature_name.startswith("supervised_"):
            return feature_name.replace("supervised_", "")
        elif feature_name.startswith("ctclip_"):
            return feature_name.replace("ctclip_", "")
        elif "_mean" in feature_name or "_max" in feature_name or "_top3" in feature_name:
            # Агрегированные фичи
            return feature_name
        else:
            return feature_name

    def unload(self):
        """Выгружает модель из памяти."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.shap_explainer is not None:
            del self.shap_explainer
            self.shap_explainer = None


class FVLMInferenceService:
    """Сервис для FVLM инференса с автоматической сегментацией."""

    def __init__(
        self,
        model_path: str,
        mae_weights_path: str,
        bert_path: str,
        config_path: str = None,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.mae_weights_path = mae_weights_path
        self.bert_path = bert_path
        self.config_path = config_path or str(
            Path(__file__).parent.parent / "fvlm" / "lavis" / "projects" / "blip" / "train" / "pretrain_ct.yaml"
        )
        self.device = device
        self.inference = None

        logger.info(f"FVLM Model: {model_path}")
        logger.info(f"MAE Weights: {mae_weights_path}")
        logger.info(f"BERT Path: {bert_path}")

    def load_model(self):
        """Загружает FVLM модель в память."""
        if self.inference is None:
            logger.info("Загружаем FVLM модель...")
            self.inference = NiftiInferenceSeparateMasks(
                model_path=self.model_path,
                mae_weights_path=self.mae_weights_path,
                bert_path=self.bert_path,
                config_path=self.config_path,
            )
            logger.info("✅ FVLM модель загружена")

    def unload_model(self):
        """Выгружает модель из памяти."""
        if self.inference is not None:
            if hasattr(self.inference, "model") and self.inference.model is not None:
                del self.inference.model
                self.inference.model = None
            del self.inference
            self.inference = None
            torch.cuda.empty_cache()
            logger.info("FVLM модель выгружена")

    def predict(self, nifti_path: str) -> Dict[str, float]:
        """Предсказание патологий для NIfTI файла."""
        if self.inference is None:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")

        logger.info(f"Запуск FVLM инференса для: {nifti_path}")
        results = self.inference.predict_single(image_path=nifti_path)
        logger.info(f"✅ FVLM инференс завершен: {len(results)} патологий")
        return results


class MedNeXtInferenceService:
    """
    Service wrapper for MedNeXt 3D classifier inference.

    Features:
    - Lazy model loading/unloading (GPU memory friendly).
    - Single-volume predict (supports .nii/.nii.gz and DICOM .zip via your helper).
    - CSV-manifest inference with resume, batching, and data_root handling.
    - Context manager support.
    """

    def __init__(self,
        device: str = "cuda",
        # Optional defaults: if omitted, your init_model() will use DEFAULT_* constants.
    ):

        self.device = device

        self.artifacts = None  # type: Optional[InferenceArtifacts]

        logger.info(f"MedNeXt device: {self.device}")

    def load_model(self) -> None:
        """
        Loads model and preprocessing pipeline into memory (idempotent).
        If custom config/weights were passed, temporarily patch DEFAULT_* before init.
        """
        if self.artifacts is not None:
            return

        logger.info("Loading MedNeXt model...")
        self.artifacts = init_model(device=self.device)

        logger.info("✅ MedNeXt model loaded")

    def unload_model(self) -> None:
        """
        Unloads model and frees GPU memory (idempotent).
        """
        if self.artifacts is None:
            return
        try:
            if hasattr(self.artifacts, "model") and self.artifacts.model is not None:
                del self.artifacts.model
        finally:
            self.artifacts = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("MedNeXt model unloaded")

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload_model()

    @property
    def classes(self):
        if self.artifacts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return list(self.artifacts.classes)


    def predict(self, volume_path: Union[str, Path]) -> Dict[str, float]:
        """
        Run inference on a single volume (NIfTI or DICOM ZIP).
        Returns: {class_name: probability}
        """
        if self.artifacts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        volume_path = str(volume_path)
        logger.info(f"Running MedNeXt inference for: {volume_path}")
        out = predict_single_volume(volume_path, self.artifacts)
        logger.info(f"✅ Inference done for {volume_path}")
        return out


class LightGBMInferenceService:
    """Сервис для полного инференса: Supervised + CT-CLIP + Diffusion + FVLM + MedNext + LightGBM."""

    def __init__(
        self,
        supervised_model_path: str,
        ctclip_model_path: str,
        lightgbm_model_path: str,
        optimal_threshold: float = 0.5,
        device: str = "cuda",
        diffusion_classifier_path: str = None,
        diffusion_unet_path: str = None,
        classifier_scale: float = 200.0,
        use_diffusion_reconstruction: bool = False, 
        # FVLM параметры
        fvlm_model_path: str = None,
        fvlm_mae_weights_path: str = None,
        fvlm_bert_path: str = None,
        fvlm_config_path: str = None,
        # MedNext
        use_mednext: bool = True,
    ):
        self.supervised_model_path = supervised_model_path
        self.ctclip_model_path = ctclip_model_path
        self.lightgbm_model_path = lightgbm_model_path
        self.device = device
        self.use_diffusion = diffusion_classifier_path is not None
        self.use_fvlm = fvlm_model_path is not None
        self.use_diffusion_reconstruction = use_diffusion_reconstruction
        self.use_mednext = use_mednext

        self.supervised_inference = SupervisedModelInference(supervised_model_path, device)
        self.ctclip_inference = UniversalCTInference(model_path=ctclip_model_path, device=device, verbose=False)
        self.lightgbm_inference = LightGBMInference(lightgbm_model_path, optimal_threshold)

        # Инициализируем diffusion модели если пути указаны
        self.diffusion_classifier_inference = None
        self.diffusion_reconstruction_inference = None

        if self.use_diffusion:
            logger.info("Инициализация diffusion моделей...")
            self.diffusion_classifier_inference = DiffusionClassifierInference(
                model_path=diffusion_classifier_path,
                device=device,
                image_size=256,
            )

            if diffusion_unet_path and self.use_diffusion_reconstruction:
                self.diffusion_reconstruction_inference = DiffusionReconstructionInference(
                    diffusion_model_path=diffusion_unet_path,
                    classifier_model=self.diffusion_classifier_inference,
                    device=device,
                    image_size=256,
                    classifier_scale=classifier_scale,
                    noise_level=500,
                    num_inference_steps=20,
                )
                logger.info("Оба diffusion inference инициализированы (classifier + reconstruction)")
            else:
                logger.info("Только diffusion classifier inference инициализирован")

        # Инициализируем FVLM если пути указаны
        self.fvlm_inference = None
        if self.use_fvlm:
            if not fvlm_mae_weights_path or not fvlm_bert_path:
                logger.warning("FVLM параметры неполные, FVLM будет отключен")
                self.use_fvlm = False
            else:
                logger.info("Инициализация FVLM...")
                self.fvlm_inference = FVLMInferenceService(
                    model_path=fvlm_model_path,
                    mae_weights_path=fvlm_mae_weights_path,
                    bert_path=fvlm_bert_path,
                    config_path=fvlm_config_path,
                    device=device,
                )
                logger.info("FVLM inference инициализирован")

        # MedNeXt
        self.mednext_inference = None
        if self.use_mednext:
            logger.info("Инициализация MedNeXt...")
            self.mednext_inference = MedNeXtInferenceService(
                device=device,
            )
            logger.info("MedNeXt inference инициализирован")


    def process_zip_archive(self, zip_path: str) -> Dict:
        """
        Обрабатывает ZIP архив с DICOM файлами.

        Returns:
            Dict с полными предсказаниями всех трех моделей
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = os.path.join(temp_dir, "dicom_extract")
            nifti_dir = os.path.join(temp_dir, "nifti")
            os.makedirs(extract_dir, exist_ok=True)
            os.makedirs(nifti_dir, exist_ok=True)

            # Извлекаем и выбираем лучшую серию
            logger.info("Извлекаем DICOM файлы...")
            series_dict = DICOMSeriesSelector.extract_series_from_zip(zip_path, extract_dir)
            logger.info(f"Найдено {len(series_dict)} серий")

            series_uid, dicom_paths = DICOMSeriesSelector.select_best_series(series_dict)
            logger.info(f"Выбрана серия: {series_uid}")

            # Получаем Study UID из первого DICOM файла
            dcm = pydicom.dcmread(dicom_paths[0], stop_before_pixels=True)
            study_uid = str(dcm.StudyInstanceUID)

            # Конвертируем выбранную серию в NIFTI
            logger.info("Конвертируем DICOM в NIFTI...")
            series_dir = os.path.join(extract_dir, "selected_series")
            os.makedirs(series_dir, exist_ok=True)

            # Копируем только выбранные файлы
            import shutil

            for dcm_path in dicom_paths:
                shutil.copy(dcm_path, series_dir)

            nifti_file = os.path.join(nifti_dir, f"{series_uid}.nii.gz")

            try:
                dicom2nifti.dicom_series_to_nifti(series_dir, nifti_file, reorient_nifti=False)
            except Exception as e:
                logger.error(f"Ошибка конвертации DICOM в NIFTI: {e}")
                raise

            # 1. FVLM модель (если включена)
            fvlm_preds = None
            if self.use_fvlm and self.fvlm_inference:
                logger.info("Загружаем FVLM модель...")
                self.fvlm_inference.load_model()
                fvlm_preds = self.fvlm_inference.predict(nifti_file)

                # Выгружаем FVLM
                self.fvlm_inference.unload_model()
                logger.info("FVLM модель выгружена")

            # TODO: skip next steps if lungs too short

            # 2. Supervised модель
            logger.info("Загружаем supervised модель...")
            self.supervised_inference.load_model()
            supervised_preds = self.supervised_inference.predict(nifti_file)
            self.supervised_inference.unload()
            logger.info("Supervised модель выгружена")

            # 3. CT-CLIP модель
            logger.info("Загружаем CT-CLIP модель...")
            ctclip_result = self.ctclip_inference.infer(nifti_file, custom_pathologies=CTCLIP_PATHOLOGIES)

            ctclip_preds = {}
            for pathology, prob in ctclip_result["pathology_predictions"].items():
                ctclip_preds[f"ctclip_{pathology}"] = prob

            # Выгружаем CT-CLIP модель
            if hasattr(self.ctclip_inference, "model") and self.ctclip_inference.model is not None:
                del self.ctclip_inference.model
                self.ctclip_inference.model = None
                torch.cuda.empty_cache()
            logger.info("CT-CLIP модель выгружена")

            # 4. MedNext
            mednext_preds = None
            if self.use_mednext and self.mednext_inference is not None:
                logger.info("Загружаем MedNeXt модель...")
                self.mednext_inference.load_model()
                # Note: MedNeXt supports both NIfTI and DICOM ZIP; here we already have NIfTI path.
                raw_probs = self.mednext_inference.predict(nifti_file)  # {class: prob}
                # Prefix features to avoid collisions and make columns explicit
                mednext_preds = {f"mednext_{cls}": float(prob) for cls, prob in raw_probs.items()}
                self.mednext_inference.unload_model()
                logger.info("MedNeXt модель выгружена")


            # 5. Diffusion модели (если включены)
            diffusion_classifier_preds = None
            diffusion_reconstruction_scores = None

            if self.use_diffusion and self.diffusion_classifier_inference:
                logger.info("Загружаем diffusion classifier...")
                self.diffusion_classifier_inference.load_model()
                diffusion_classifier_preds = self.diffusion_classifier_inference.predict(nifti_file)
                logger.info(f"Diffusion classifier predictions: {diffusion_classifier_preds}")

                if self.diffusion_reconstruction_inference and self.use_diffusion_reconstruction:
                    logger.info("Загружаем diffusion reconstruction...")
                    self.diffusion_reconstruction_inference.load_model()
                    diffusion_reconstruction_scores = self.diffusion_reconstruction_inference.predict(nifti_file)
                    logger.info(f"Diffusion reconstruction scores: {diffusion_reconstruction_scores}")

                    # Выгружаем reconstruction UNet (classifier остается)
                    if self.diffusion_reconstruction_inference.diffusion_model:
                        del self.diffusion_reconstruction_inference.diffusion_model
                        del self.diffusion_reconstruction_inference.diffusion
                        self.diffusion_reconstruction_inference.diffusion_model = None
                        self.diffusion_reconstruction_inference.diffusion = None
                        torch.cuda.empty_cache()
                    logger.info("Diffusion reconstruction модель выгружена")

                # Выгружаем classifier
                if self.diffusion_classifier_inference.model:
                    del self.diffusion_classifier_inference.model
                    self.diffusion_classifier_inference.model = None
                    torch.cuda.empty_cache()
                logger.info("Diffusion classifier выгружен")

            # 6. LightGBM модель
            logger.info("Загружаем LightGBM модель...")
            self.lightgbm_inference.load_model()

            # Подготавливаем фичи для LightGBM (включая diffusion если есть)
            features_df = self.lightgbm_inference.prepare_features(
                supervised_preds,
                ctclip_preds,
                diffusion_classifier_preds,
                diffusion_reconstruction_scores,
                mednext_preds=mednext_preds,
            )

            # Делаем предсказание
            lgbm_result = self.lightgbm_inference.predict(features_df)

            # Выгружаем LightGBM
            self.lightgbm_inference.unload()
            logger.info("LightGBM модель выгружена")

            # Формируем финальный результат
            result = {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "probability_of_pathology": lgbm_result["probability"],
                "pathology": int(lgbm_result["prediction"]),
                "most_dangerous_pathology_type": lgbm_result["most_dangerous_pathology"],
            }

            return result


