#!/usr/bin/env python3
"""
FastAPI приложение для CT-CLIP + LightGBM инференса
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

from api.inference_service import LightGBMInferenceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CT-CLIP + LightGBM Inference API",
    description="API для инференса CT-CLIP моделей с LightGBM классификацией на DICOM архивах",
    version="2.0.0",
)

SUPERVISED_MODEL_PATH = os.getenv("SUPERVISED_MODEL_PATH", "/app/models/supervised_model.pt")
CTCLIP_MODEL_PATH = os.getenv("CTCLIP_MODEL_PATH", "/app/models/CT_VocabFine_v2.pt")
LIGHTGBM_MODEL_PATH = os.getenv("LIGHTGBM_MODEL_PATH", "/app/models/lightgbm_model.pkl")
OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", "0.4"))
DEVICE = os.getenv("DEVICE", "cuda")

DIFFUSION_CLASSIFIER_PATH = os.getenv("DIFFUSION_CLASSIFIER_PATH", None)
DIFFUSION_UNET_PATH = os.getenv("DIFFUSION_UNET_PATH", None)
CLASSIFIER_SCALE = float(os.getenv("CLASSIFIER_SCALE", "100.0"))

# Пути к моделям FVLM
FVLM_MODEL_PATH = os.getenv("FVLM_MODEL_PATH", "/app/models/fvlm/fvlm_model.pth")
FVLM_MAE_WEIGHTS_PATH = os.getenv("FVLM_MAE_WEIGHTS_PATH", "/app/models/fvlm/mae_pretrain_vit_base.pth")
FVLM_BERT_PATH = os.getenv("FVLM_BERT_PATH", "/app/models/fvlm/BiomedVLP-CXR-BERT-specialized")
FVLM_CONFIG_PATH = os.getenv("FVLM_CONFIG_PATH", None)

inference_service = None


class InferenceResponse(BaseModel):
    """Модель ответа API."""

    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int  # 0 (норма) или 1 (патология)
    most_dangerous_pathology_type: str
    processing_status: str
    time_of_processing: float


@app.on_event("startup")
async def startup_event():
    """Инициализация сервиса при запуске."""
    global inference_service

    logger.info("Инициализация CT-CLIP + Diffusion + LightGBM Inference Service...")
    logger.info(f"Supervised Model: {SUPERVISED_MODEL_PATH}")
    logger.info(f"CT-CLIP Model: {CTCLIP_MODEL_PATH}")
    logger.info(f"LightGBM Model: {LIGHTGBM_MODEL_PATH}")
    logger.info(f"Optimal Threshold: {OPTIMAL_THRESHOLD}")
    logger.info(f"Device: {DEVICE}")

    if DIFFUSION_CLASSIFIER_PATH:
        logger.info(f"Diffusion Classifier: {DIFFUSION_CLASSIFIER_PATH}")
    if DIFFUSION_UNET_PATH:
        logger.info(f"Diffusion UNet: {DIFFUSION_UNET_PATH}")
        logger.info(f"Classifier Scale: {CLASSIFIER_SCALE}")

    if not os.path.exists(SUPERVISED_MODEL_PATH):
        raise FileNotFoundError(f"Supervised модель не найдена: {SUPERVISED_MODEL_PATH}")
    if not os.path.exists(CTCLIP_MODEL_PATH):
        raise FileNotFoundError(f"CT-CLIP модель не найдена: {CTCLIP_MODEL_PATH}")
    if not os.path.exists(LIGHTGBM_MODEL_PATH):
        raise FileNotFoundError(f"LightGBM модель не найдена: {LIGHTGBM_MODEL_PATH}")

    if DIFFUSION_CLASSIFIER_PATH and not os.path.exists(DIFFUSION_CLASSIFIER_PATH):
        raise FileNotFoundError(f"Diffusion classifier не найден: {DIFFUSION_CLASSIFIER_PATH}")
    if DIFFUSION_UNET_PATH and not os.path.exists(DIFFUSION_UNET_PATH):
        raise FileNotFoundError(f"Diffusion UNet не найден: {DIFFUSION_UNET_PATH}")

    inference_service = LightGBMInferenceService(
        supervised_model_path=SUPERVISED_MODEL_PATH,
        ctclip_model_path=CTCLIP_MODEL_PATH,
        lightgbm_model_path=LIGHTGBM_MODEL_PATH,
        optimal_threshold=OPTIMAL_THRESHOLD,
        device=DEVICE,
        diffusion_classifier_path=DIFFUSION_CLASSIFIER_PATH,
        diffusion_unet_path=DIFFUSION_UNET_PATH,
        classifier_scale=CLASSIFIER_SCALE,
        # FVLM параметры
        fvlm_model_path=FVLM_MODEL_PATH if os.path.exists(FVLM_MODEL_PATH) else None,
        fvlm_mae_weights_path=FVLM_MAE_WEIGHTS_PATH if os.path.exists(FVLM_MAE_WEIGHTS_PATH) else None,
        fvlm_bert_path=FVLM_BERT_PATH if os.path.exists(FVLM_BERT_PATH) else None,
        fvlm_config_path=FVLM_CONFIG_PATH,
    )

    logger.info("✅ Сервис инициализирован успешно!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "CT-CLIP + LightGBM Inference API",
        "version": "2.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """
    Эндпоинт для инференса на DICOM архиве.

    Args:
        file: ZIP архив с DICOM файлами

    Returns:
        InferenceResponse с предсказаниями обеих моделей
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Поддерживаются только ZIP архивы с DICOM файлами")

    start_time = time.time()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"Обрабатываем файл: {file.filename} ({len(content)} bytes)")

        result = inference_service.process_zip_archive(temp_file_path)

        os.unlink(temp_file_path)

        processing_time = time.time() - start_time
        logger.info(f"✅ Обработка завершена за {processing_time:.2f} сек")

        return InferenceResponse(
            study_uid=result["study_uid"],
            series_uid=result["series_uid"],
            probability_of_pathology=result["probability_of_pathology"],
            pathology=result["pathology"],
            most_dangerous_pathology_type=result["most_dangerous_pathology_type"],
            processing_status="Success",
            time_of_processing=processing_time,
        )

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке файла {file.filename}: {e}")
        processing_time = time.time() - start_time

        try:
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)
        except Exception:
            pass

        # Проверяем специфичные ошибки
        error_message = str(e)
        status_code = 500
        
        if "Lungs not found" in error_message or "too low relative percentage" in error_message:
            status_code = 422  # Unprocessable Entity
            error_message = "Легкие не найдены на изображении или их относительный процент слишком мал."
            logger.warning(f"⚠️ Легкие не найдены: {file.filename}")
        elif "too small" in error_message and "mm" in error_message:
            status_code = 422  # Unprocessable Entity
            error_message = f"Длина легких слишком мала. {error_message}"
            logger.warning(f"⚠️ Короткие легкие: {file.filename}")
        elif "Не найдено серий с минимальным количеством валидных срезов" in error_message:
            status_code = 422  # Unprocessable Entity
            error_message = "Недостаточно срезов для обработки (требуется минимум 20 валидных срезов)."
            logger.warning(f"⚠️ Недостаточно срезов: {file.filename}")

        return JSONResponse(
            status_code=status_code,
            content={
                "study_uid": "",
                "series_uid": "",
                "probability_of_pathology": None,
                "pathology": None,
                "most_dangerous_pathology_type": None,
                "processing_status": "Failure",
                "time_of_processing": processing_time,
                "error": error_message,
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
