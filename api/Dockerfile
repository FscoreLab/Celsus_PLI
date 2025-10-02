FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements
COPY api/requirements.txt /app/api/requirements.txt

# Устанавливаем Python зависимости с кешированием
RUN --mount=type=cache,target=/root/.cache pip3 install -r /app/api/requirements.txt
RUN pip install -U pip setuptools wheel setuptools_scm
RUN pip install --no-cache-dir "git+https://github.com/MIC-DKFZ/nnUNet@v2.6.2"
RUN pip install TotalSegmentator==2.11.0

# Скачиваем модели TotalSegmentator заранее
RUN python3 -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(291)"
RUN python3 -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(292)"
RUN python3 -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(293)"
RUN python3 -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(294)"
RUN python3 -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(295)"
RUN python3 -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(298)"


# Устанавливаем дополнительные зависимости
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip3 install beartype==0.21.0 blobfile==3.1.0

# Копируем модели из best_models
COPY best_models/supervised_model.pt /app/best_models/
COPY best_models/CT_VocabFine_v2.pt /app/best_models/

# Копируем код пакетов
COPY fvlm/ /app/fvlm/
COPY mednext/ /app/mednext/
RUN pip3 install -e /app/fvlm

COPY best_models/lightgbm_model.pkl /app/best_models/

COPY CT_CLIP/ /app/CT_CLIP/
COPY transformer_maskgit/ /app/transformer_maskgit/
COPY diffusion_anomaly/ /app/diffusion_anomaly/
COPY scripts/ /app/scripts/
COPY api/ /app/api/
COPY ct_clip_classifier.py /app/
COPY api/ctclip_pathology_groups.py /app/

# Порт для API
EXPOSE 8000

# Переменные окружения
ENV SUPERVISED_MODEL_PATH=/app/best_models/supervised_model.pt
ENV CTCLIP_MODEL_PATH=/app/best_models/CT_VocabFine_v2.pt
ENV LIGHTGBM_MODEL_PATH=/app/best_models/lightgbm_model.pkl
ENV OPTIMAL_THRESHOLD=0.57

# FVLM пути
ENV FVLM_MODEL_PATH=/app/fvlm/weights/fvlm_model.pth
ENV FVLM_MAE_WEIGHTS_PATH=/app/fvlm/weights/mae_pretrain_vit_base.pth
ENV FVLM_BERT_PATH=/app/fvlm/weights/BiomedVLP-CXR-BERT-specialized

ENV DEVICE=cuda
ENV PYTHONPATH=/app:$PYTHONPATH

# Запуск API
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600", "--timeout-graceful-shutdown", "600"]
