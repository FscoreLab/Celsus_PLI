#!/usr/bin/env python3
"""
Скрипт для инференса модели FVLM на NIfTI объемах с отдельными масками органов
Оптимизированная версия для работы с ограниченной GPU памятью
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nibabel as nib
from monai import transforms
from typing import List, Tuple, Dict
import gc
import json
import glob
from .utils import speedup_numpy_unique_cpu, get_bbox_slices

from lavis.common.config import Config
from lavis.common.registry import registry
from .lct_validation import LungValidator

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map


def masks_to_boxes_3d(masks):
    """Вычисляет ограничивающие боксы для 3D масок"""
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


def center_crop(image, mask, crop_size):
    """Центральная обрезка изображения и маски"""
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()
    
    crop_d, crop_h, crop_w = max(crop_size[0], z_max - z_min), \
                            max(crop_size[1], y_max - y_min), \
                            max(crop_size[2], x_max - x_min)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    
    d, h, w = image.shape[-3:]

    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    
    return image[..., z_start:z_end, y_start:y_end, x_start:x_end], \
           mask[..., z_start:z_end, y_start:y_end, x_start:x_end]


def clear_gpu_memory():
    """Очищает GPU память"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def save_multiclass_mask(mask_array: np.ndarray, reference_img: nib.Nifti1Image, output_path: str):
    """Сохраняет мультикласс маску в файл NIfTI"""
    mask_img = nib.Nifti1Image(mask_array, reference_img.affine, reference_img.header)
    nib.save(mask_img, output_path)
    print(f"✓ Маска сохранена: {output_path}")


class NiftiInferenceSeparateMasks:
    """Класс для инференса на NIfTI объемах с отдельными масками органов"""
    
    def __init__(self, model_path: str, mae_weights_path: str, bert_path: str, config_path: str = None):
        # Только 3 органа (без пищевода)
        self.organs = ['lung', 'heart', 'esophagus', 'aorta']
        
        # Убираем задачу с пищеводом
        self.test_items = [
            ('lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'),
            ('lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'), 
            ('lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'),
            ('lung', 'Lung opacity', 'Not Opacity.', 'Opacity.'),
            ('lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'),
            ('lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'),
            ('lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'),
            ('lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'),
            ('lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'),
            ('lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'),
            ('lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'),
            ('heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'),
            ('heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'),
            ('heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.', 'Coronary artery wall calcification.'),
            ('aorta', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'),
            ('lung', 'Pneumothorax', 'No Pneumothorax.', 'Pneumothorax.'),  # new
            ('aorta', 'Aortic aneurysm', 'No Aortic aneurysm.', 'Aortic aneurysm.'),  # new
            ('lung', 'Lung cancer', 'Not Lung cancer.', 'Lung cancer.'),  # new
        ]
        
        self.test_organs = list(set([item[0] for item in self.test_items]))

        base_load_transforms = [
            transforms.LoadImaged(keys=["arr"], image_only=True, ensure_channel_first=True),
            transforms.Transposed(keys=["arr"], indices=(0, 3, 2, 1))
        ]

        self.base_loader = transforms.Compose(base_load_transforms)
        
        # Настройка трансформов для загрузки данных
        self.image_loader = transforms.Compose([
            *base_load_transforms,
            transforms.ScaleIntensityRanged(
                keys=["arr"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
            )
        ])
        
        self.pad_func = transforms.DivisiblePadd(
            keys=["image", "label"], 
            k=(16, 16, 32),
            mode='constant', 
            constant_values=0,
            method="end"
        )
        
        # Очищаем память перед загрузкой модели
        clear_gpu_memory()
        
        # Загрузка модели
        self.model = self._load_model(model_path, mae_weights_path, bert_path, config_path)
        self.text_feat_dict = self.model.prepare_text_feat(self.test_items)
        
        # Очищаем память после загрузки
        clear_gpu_memory()
        
    def _load_model(self, model_path: str, mae_weights_path: str, bert_path: str, config_path: str = None):
        """Загружает предобученную модель"""
        if config_path is None:
            config_path = 'lavis/projects/blip/train/pretrain_ct.yaml'
        
        # Создаем временный конфиг с правильными путями
        args = argparse.Namespace(cfg_path=config_path, options=[
            f"model.med_config_path={os.path.join(bert_path, 'config.json')}"
        ])
        cfg = Config(args)
        
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config, mae_weights_path)
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location='cpu')
            model.load_state_dict(ckpt['model'], strict=False)
            print(f"Модель загружена из {model_path}")
        else:
            print(f"Предупреждение: файл {model_path} не найден")
        
        model.eval()
        
        # Используем mixed precision для экономии памяти
        if torch.cuda.is_available():
            model = model.half()  # Конвертируем в FP16
        
        model.cuda()
        
        return model
    

    def generate_multiclass_mask_from_totalseg(self, image_path: str) -> np.ndarray:
        """
        Генерирует мультикласс маску напрямую из TotalSegmentator без промежуточных файлов

        Args:
            image_path: Путь к NIfTI изображению

        Returns:
            numpy array с мультикласс маской (lung=1, heart=2, esophagus=3, aorta=4)
        """

        print(f"Генерация мультикласс маски для: {image_path}")

        # Загружаем изображение
        input_img = nib.load(image_path)
        print(f"Размер изображения: {input_img.shape}")

        spacing = input_img.header.get_zooms()
        
        # # Ресайз к спейсингу (6,6,6) перед TotalSegmentator
        # original_spacing = np.abs(input_img.header.get_zooms()[:3])  # Получаем спейсинг из заголовка
        # target_spacing = np.array([6.0, 6.0, 6.0])
        #
        # print(f"Исходный спейсинг: {original_spacing}")
        # print(f"Целевой спейсинг: {target_spacing}")
        #
        # # Вычисляем новый размер на основе соотношения спейсингов
        # original_size = np.array(input_img.shape)
        # scale_factor = original_spacing / target_spacing
        # target_size = (original_size * scale_factor).astype(int)
        #
        # print(f"Исходный размер: {original_size}")
        # print(f"Целевой размер: {target_size}")
        #
        # # Получаем данные изображения и делаем ресайз
        # input_arr = input_img.get_fdata()
        #
        # # Добавляем batch и channel dimensions для interpolate
        # input_tensor = torch.tensor(input_arr, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
        #
        # # Ресайз с помощью trilinear интерполяции
        # resized_tensor = torch.nn.functional.interpolate(
        #     input=input_tensor,
        #     size=tuple(target_size),
        #     mode="trilinear",
        #     align_corners=False
        # )
        #
        # # Убираем batch и channel dimensions
        # resized_arr = resized_tensor.squeeze(0).squeeze(0).numpy()
        #
        # # Обновляем аффинную матрицу для нового спейсинга
        # new_affine = input_img.affine.copy()
        # # Присваиваем новые значения спейсинга диагональным элементам
        # for i in range(3):
        #     # Сохраняем знак (направление оси) и устанавливаем новый спейсинг
        #     sign = np.sign(new_affine[i, i])
        #     new_affine[i, i] = sign * target_spacing[i]
        #
        # # Создаем новое NIfTI изображение с ресайзнутыми данными
        # resized_img = nib.Nifti1Image(resized_arr, new_affine, input_img.header)
        
        # print(f"✓ Ресайз завершен: {resized_arr.shape}")

        totalseg_id_class_map = class_map["total"]

        # Маппинг ID органов TotalSegmentator -> наши ID
        totalseg_to_our_mapping = {
            # Легкие (все доли) -> ID 1
            10: 1,  # lung_upper_lobe_left
            11: 1,  # lung_lower_lobe_left
            12: 1,  # lung_upper_lobe_right
            13: 1,  # lung_middle_lobe_right
            14: 1,  # lung_lower_lobe_right
            # Сердце -> ID 2
            51: 2,  # heart
            # Пищевод -> ID 3
            #15: 3,  # esophagus
            # Аорта -> ID 4
            52: 4,  # aorta
        }

        # Запускаем TotalSegmentator - возвращает NIfTI изображение с мультилейбел маской
        print("Запуск TotalSegmentator...")
        segmentation_img = totalsegmentator(input_img, quiet=False, ml=True, fast=True,
                                            roi_subset=[totalseg_id_class_map[id_] for id_ in totalseg_to_our_mapping])

        # Получаем массив данных
        segmentation_array = segmentation_img.get_fdata()
        print(f"Размер сегментации: {segmentation_array.shape}")

        #  # Добавляем batch и channel dimensions для interpolate
        # seg_tensor = torch.tensor(segmentation_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        #
        # # Обратный ресайз с помощью nearest neighbor интерполяции
        # original_size_seg = torch.nn.functional.interpolate(
        #     input=seg_tensor,
        #     size=tuple(original_size.astype(int)),
        #     mode="nearest"
        # )
        #
        # # Убираем batch и channel dimensions
        # segmentation_array = original_size_seg.squeeze(0).squeeze(0).numpy()
        #
        # print(f"✓ Обратный ресайз завершен: {segmentation_array.shape}")
        # print(f"Уникальные значения в ресайзнутой сегментации: {np.unique(segmentation_array)}")

        # Создаем мультикласс маску для наших 4 органов
        multiclass_mask = np.zeros_like(segmentation_array, dtype=np.uint8)
        organs_found = []

        # Применяем маппинг векторизованно
        unique_values = speedup_numpy_unique_cpu(segmentation_array)
        print(f"Уникальные значения в сегментации: {unique_values}")
        for totalseg_id, our_id in totalseg_to_our_mapping.items():
            if totalseg_id in unique_values:
                organ_mask = segmentation_array == totalseg_id
                multiclass_mask[organ_mask] = our_id
                
                organ_names = {1: 'lung', 2: 'heart', 3: 'esophagus', 4: 'aorta'}
                organ_name = organ_names.get(our_id, f'organ_{our_id}')
                
                if organ_name not in organs_found:
                    organs_found.append(organ_name)
                    voxel_count = np.sum(organ_mask)
                    print(f"✓ Найден {organ_name}: {voxel_count} вокселей (ID {totalseg_id} -> {our_id})")

        print(f"✓ Обработано органов: {', '.join(organs_found)}")
        #print(f"✓ Финальные ID в маске: {np.unique(multiclass_mask)}")

        return multiclass_mask, spacing

    def preprocess_nifti_with_mask_array(self, image_path: str, mask_array: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Предобработка NIfTI файла с готовым массивом маски"""
        
        print(f"Предобработка изображения: {image_path}")
        
        # Загружаем изображение
        image_data = self.image_loader({"arr": image_path})
        image = image_data["arr"]
        
        # Конвертируем массив маски в тензор
        mask = torch.from_numpy(mask_array).float().unsqueeze(0)  # Add channel dim
        
        print(f"Размер изображения: {image.shape}")
        print(f"Размер маски: {mask.shape}")
        
        # Обрезка ROI с отступами
        mask_np = mask[0].numpy()
        roi_coords = np.nonzero(mask_np)
        
        if len(roi_coords[0]) == 0:
            raise ValueError("Маска пустая - нет ROI для обработки")
        
        min_dhw = torch.from_numpy(np.array([np.min(coord) for coord in roi_coords]))
        max_dhw = torch.from_numpy(np.array([np.max(coord) for coord in roi_coords]))

        extend_d, extend_hw = 5, 20
        min_dhw = torch.maximum(
            min_dhw - torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([0, 0, 0]),
        )
        max_dhw = torch.minimum(
            max_dhw + torch.tensor([extend_d, extend_hw, extend_hw]),
            torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
        )

        print(f"ROI координаты: {min_dhw.tolist()} -> {max_dhw.tolist()}")

        image = image[:, min_dhw[0]:max_dhw[0], min_dhw[1]:max_dhw[1], min_dhw[2]:max_dhw[2]]
        mask = mask[:, min_dhw[0]:max_dhw[0], min_dhw[1]:max_dhw[1], min_dhw[2]:max_dhw[2]]
        
        # Паддинг до фиксированного размера
        padder = transforms.Compose([
            transforms.SpatialPadd(
                keys=["image", "label"],
                spatial_size=(112, 256, 352),
                mode="constant",
                constant_values=0
            )
        ])
        
        padded_data = padder({"image": image, "label": mask})
        
        print(f"Финальные размеры: image={padded_data['image'].shape}, mask={padded_data['label'].shape}")
        
        return padded_data["image"], padded_data["label"]
    
    @torch.inference_mode()
    def predict_single(self, image_path: str, save_mask_path: str = None) -> Dict[str, float]:
        """
        Предсказание для одного NIfTI объема

        Args:
            image_path: Путь к NIfTI изображению
            save_mask_path: Путь для сохранения сгенерированной маски
        """

        print("Генерация маски с помощью TotalSegmentator...")
        mask_array, spacing = self.generate_multiclass_mask_from_totalseg(image_path)

        # Сохраняем маску, если указан путь
        if save_mask_path:
            reference_img = nib.load(image_path)
            save_mask_path = os.path.join(save_mask_path, os.path.basename(image_path))
            save_multiclass_mask(mask_array, reference_img, save_mask_path)

        mask_array = transforms.Transposed(keys=["arr"], indices=(0, 3, 2, 1))({"arr": mask_array[None]})["arr"][0].numpy()

        print(f"✓ Маска готова, размер: {mask_array.shape}")

        validator = LungValidator()
        validator.validate_lung_length(mask_array == 1, spacing[-1]) # TODO: check spacing order

        # Предобработка
        image, mask = self.preprocess_nifti_with_mask_array(image_path, mask_array)

        # Определяем доступные органы на CPU
        whole_organ_sizes = {}
        available_organs = []

        organ_id_mapping = {'lung': 1, 'heart': 2, 'esophagus': 3, 'aorta': 4}

        for organ, organ_id in organ_id_mapping.items():
            organ_size = torch.eq(mask, organ_id).sum().item()
            if organ_size > 0:
                whole_organ_sizes[organ] = organ_size
                available_organs.append(organ)

        print(f"Найденные органы: {available_organs}")

        if not available_organs:
            print("Предупреждение: в маске не найдено ни одного органа")
            return {}

        # Фильтруем тестовые элементы по доступным органам
        test_items = [item for item in self.test_items if item[0] in available_organs]

        if not test_items:
            print("Предупреждение: нет тестовых элементов для доступных органов")
            return {}

        # ROI size для скользящего окна
        roi_size = (112, 288, 352)
        results = {}

        # Группируем тестовые элементы по органам
        organ_test_items = {}
        for test_item in test_items:
            organ_name = test_item[0]
            if organ_name not in organ_test_items:
                organ_test_items[organ_name] = []
            organ_test_items[organ_name].append(test_item)

        print(f"Обработка по органам: {list(organ_test_items.keys())}")

        # Обработка каждого органа отдельно для экономии памяти
        for organ_name, organ_tasks in organ_test_items.items():
            organ_id = organ_id_mapping[organ_name]

            print(f"Обработка органа {organ_name}: {len(organ_tasks)} патологий")

            # Очищаем память перед обработкой каждого органа
            clear_gpu_memory()

            bbox_slices = get_bbox_slices(mask[0] == organ_id, 3, 0.1)
            window_patch = image.array[0][bbox_slices][None]
            window_mask = (mask.array[0][bbox_slices][None] == organ_id).astype(np.uint8)

            window_patch = torch.nn.functional.interpolate(
                input=torch.tensor(window_patch[None]), size=roi_size, mode="trilinear",
            )
            window_mask = torch.nn.functional.interpolate(
                input=torch.tensor(window_mask[None]), size=roi_size, mode="nearest",
            )

            window_mask[window_mask == 1] = organ_id  # for strange logic in self.model.forward_test_win

            # # Переносим данные в GPU только когда нужно
            # image_gpu = image.unsqueeze(0).cuda()
            # mask_gpu = mask.unsqueeze(0).cuda()
            #
            # # Конвертируем в FP16 для экономии памяти
            # if torch.cuda.is_available():
            #     image_gpu = image_gpu.half()
            #     mask_gpu = mask_gpu.half()
            #
            # # Центральная обрезка для текущего органа
            # window_patch, window_mask = center_crop(
            #     image_gpu,
            #     torch.eq(mask_gpu, organ_id),
            #     crop_size=roi_size
            # )
            # window_mask = window_mask.float()
            # window_mask[window_mask == 1] = organ_id
            #
            # # Освобождаем память от полных изображений
            # del image_gpu, mask_gpu
            # clear_gpu_memory()

            # Паддинг до кратного размера
            pad_data = self.pad_func({'image': window_patch[0].cpu(), 'label': window_mask[0].cpu()})
            window_patch, window_mask = pad_data['image'], pad_data['label']

            # Переносим обратно в GPU для инференса
            window_patch = window_patch.cuda()
            window_mask = window_mask.cuda()

            if torch.cuda.is_available():
                window_patch = window_patch.half()
                window_mask = window_mask.half()

            # Создаем словарь для ВСЕХ задач этого органа
            temp_organ_logits = dict(zip(organ_tasks, [[] for _ in organ_tasks]))

            # Инференс через модель - ОДИН раз для всех патологий органа
            temp_organ_logits = self.model.forward_test_win(
                window_patch[None],
                window_mask[None],
                temp_organ_logits,
                [organ_name],  # Только текущий орган
                self.text_feat_dict,
                {},
                {organ_name: whole_organ_sizes[organ_name]},
                skip_organ=list(self.organs).index(organ_name) if organ_name in self.organs else None
            )

            # Обработка результатов для всех патологий этого органа
            for item, probs in temp_organ_logits.items():
                if len(probs) > 0:
                    avg_prob = np.concatenate(probs).mean(0)[1]
                    pathology_name = item[1]
                    results[pathology_name] = float(avg_prob)

            print(f"✓ Обработано {len(organ_tasks)} патологий для органа {organ_name}")

            # Освобождаем память после обработки органа
            del window_patch, window_mask, temp_organ_logits
            clear_gpu_memory()

        return results


def save_results_to_json(results_dict: Dict, output_path: str):
    """Сохраняет результаты в JSON файл"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Результаты сохранены в: {output_path}")


def print_results_summary(results: Dict[str, float], filename: str, test_items: List):
    """Выводит краткую сводку результатов для файла"""
    print(f"\nРезультаты для {filename}:")
    print("="*50)
    
    # Группируем по органам
    lung_results = {k: v for k, v in results.items() if any(item[0] == 'lung' and item[1] == k for item in test_items)}
    heart_results = {k: v for k, v in results.items() if any(item[0] == 'heart' and item[1] == k for item in test_items)}
    esophagus_results = {k: v for k, v in results.items() if any(item[0] == 'esophagus' and item[1] == k for item in test_items)}
    aorta_results = {k: v for k, v in results.items() if any(item[0] == 'aorta' and item[1] == k for item in test_items)}
    
    if lung_results:
        print(f"ЛЕГКИЕ ({len(lung_results)} патологий):")
        for pathology, prob in lung_results.items():
            print(f"  {pathology}: {prob:.4f}")
    
    if heart_results:
        print(f"\nСЕРДЦЕ ({len(heart_results)} патологий):")
        for pathology, prob in heart_results.items():
            print(f"  {pathology}: {prob:.4f}")
    
    if esophagus_results:
        print(f"\nПИЩЕВОД ({len(esophagus_results)} патологий):")
        for pathology, prob in esophagus_results.items():
            print(f"  {pathology}: {prob:.4f}")
    
    if aorta_results:
        print(f"\nАОРТА ({len(aorta_results)} патологий):")
        for pathology, prob in aorta_results.items():
            print(f"  {pathology}: {prob:.4f}")
    
    print(f"\nВсего проанализировано: {len(results)} патологий")


def main():
    parser = argparse.ArgumentParser(description="Инференс FVLM модели на NIfTI объемах")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к файлу модели (.pth)')
    parser.add_argument('--mae_weights_path', type=str, required=True,
                       help='Путь к предобученным MAE весам')
    parser.add_argument('--bert_path', type=str, required=True,
                       help='Путь к директории с BiomedVLP-CXR-BERT-specialized')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Путь к конфигурационному файлу')
    
    # Изменяем input_path для поддержки датафрейма
    parser.add_argument('--input_path', type=str, required=True,
                       help='Путь к NIfTI изображению, папке с изображениями или CSV файлу с колонкой study_file')
    parser.add_argument('--csv_column', type=str, default='study_file',
                       help='Название колонки в CSV файле с путями к NIfTI файлам (по умолчанию: study_file)')
    
    # Опции для масок
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument('--separate_masks', action='store_true',
                           help='Использовать отдельные маски органов')
    mask_group.add_argument('--use_totalsegmentator', action='store_true',
                           help='Использовать TotalSegmentator для генерации масок')
    
    # Опции для сохранения
    parser.add_argument('--save_mask_path', type=str,
                       help='Путь для сохранения сгенерированной маски Totalsegmentator')
    parser.add_argument('--output_json', type=str, default='results.json',
                       help='Путь для сохранения результатов в JSON формате (по умолчанию: results.json)')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод результатов для каждого файла')

    args = parser.parse_args()

    if args.save_mask_path:
        os.makedirs(args.save_mask_path, exist_ok=True)
    
    # Определение источника NIfTI файлов
    if os.path.isfile(args.input_path):
        if args.input_path.endswith('.csv'):
            # Работа с датафреймом
            print(f"Загрузка датафрейма из: {args.input_path}")
            try:
                df = pd.read_csv(args.input_path)
                if args.csv_column not in df.columns:
                    print(f"Ошибка: колонка '{args.csv_column}' не найдена в CSV файле")
                    print(f"Доступные колонки: {list(df.columns)}")
                    return
                
                nifti_files = df[args.csv_column].dropna().tolist()
                print(f"Найдено {len(nifti_files)} файлов в датафрейме")
                
                # Проверяем существование файлов
                existing_files = []
                missing_files = []
                for file_path in nifti_files:
                    if os.path.exists(file_path):
                        existing_files.append(file_path)
                    else:
                        missing_files.append(file_path)
                
                if missing_files:
                    print(f"Предупреждение: {len(missing_files)} файлов не найдено:")
                    for missing_file in missing_files[:5]:  # Показываем только первые 5
                        print(f"  {missing_file}")
                    if len(missing_files) > 5:
                        print(f"  ... и еще {len(missing_files) - 5} файлов")
                
                nifti_files = existing_files
                print(f"Будет обработано {len(nifti_files)} существующих файлов")
                
            except Exception as e:
                print(f"Ошибка при загрузке CSV файла: {str(e)}")
                return
        elif args.input_path.endswith('.nii.gz'):
            # Один NIfTI файл
            nifti_files = [args.input_path]
        else:
            print("Ошибка: неподдерживаемый формат файла. Ожидается .nii.gz или .csv")
            return
    elif os.path.isdir(args.input_path):
        # Папка с NIfTI файлами
        nifti_files = glob.glob(os.path.join(args.input_path, "*.nii.gz"))
    else:
        print(f"Ошибка: путь не найден: {args.input_path}")
        return
    
    if not nifti_files:
        print("Ошибка: не найдено ни одного NIfTI файла для обработки")
        return

    # Проверяем существование файлов
    required_files = [
        (args.mae_weights_path, "MAE веса"),
        (os.path.join(args.bert_path, 'config.json'), "BiomedVLP-CXR-BERT config.json"),
        (os.path.join(args.bert_path, 'pytorch_model.bin'), "BiomedVLP-CXR-BERT pytorch_model.bin"),
    ]
    
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"Ошибка: {description} не найден: {file_path}")
            if "MAE" in description:
                print("Скачайте: wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth")
            elif "BiomedVLP" in description:
                print("Скачайте: git clone https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized")
            return
    
    # Выводим информацию о GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Доступная GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Инициализируем инференс
    print("Инициализация модели...")
    inference = NiftiInferenceSeparateMasks(args.model_path, args.mae_weights_path, args.bert_path, args.config_path)
    
    # Обработка всех файлов
    all_results = {}
    failed_files = []
    
    print(f"\nНачинаем обработку {len(nifti_files)} файлов...")
    
    for i, image_path in enumerate(tqdm(nifti_files, desc="Обработка файлов")):
        tmp_dir = "/mnt/data4tb/Totalsegmenator_preds/CT_RATE_valid"

        filename = os.path.basename(image_path)
        if os.path.exists(os.path.join(tmp_dir, filename)):
            continue
        print(f"\n[{i+1}/{len(nifti_files)}] Обработка: {filename}")
        
        try:
            results = inference.predict_single(
                image_path,
                save_mask_path=args.save_mask_path
            )

            all_results[filename] = results

            save_results_to_json(all_results, args.output_json)  # for safety

            # Выводим результаты, если включен verbose режим
            if args.verbose:
                print_results_summary(results, filename, inference.test_items)
            else:
                print(f"✓ Обработано {len(results)} патологий")
                
        except Exception as e:
            print(f"✗ Ошибка при обработке {filename}: {str(e)}")
            failed_files.append((filename, str(e)))
            continue
    
    # Сохраняем результаты в JSON
    if all_results:
        save_results_to_json(all_results, args.output_json)
        
        # Выводим общую статистику
        print(f"\n{'='*60}")
        print(f"ОБЩАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"Успешно обработано: {len(all_results)} файлов")
        print(f"Ошибок: {len(failed_files)} файлов")
        
        if failed_files:
            print(f"\nФайлы с ошибками:")
            for filename, error in failed_files:
                print(f"  {filename}: {error}")
        
        # Статистика по патологиям
        if all_results:
            sample_result = next(iter(all_results.values()))
            print(f"\nПатологии ({len(sample_result)} штук):")
            
            # Группируем по органам для статистики
            lung_pathologies = [k for k in sample_result.keys() if any(item[0] == 'lung' and item[1] == k for item in inference.test_items)]
            heart_pathologies = [k for k in sample_result.keys() if any(item[0] == 'heart' and item[1] == k for item in inference.test_items)]
            esophagus_pathologies = [k for k in sample_result.keys() if any(item[0] == 'esophagus' and item[1] == k for item in inference.test_items)]
            aorta_pathologies = [k for k in sample_result.keys() if any(item[0] == 'aorta' and item[1] == k for item in inference.test_items)]
            
            print(f"  Легкие: {len(lung_pathologies)} патологий")
            print(f"  Сердце: {len(heart_pathologies)} патологий")
            print(f"  Пищевод: {len(esophagus_pathologies)} патологий")
            print(f"  Аорта: {len(aorta_pathologies)} патологий")
    
    else:
        print("Не удалось обработать ни одного файла")


if __name__ == '__main__':
    main()
