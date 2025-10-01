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

# Устанавливаем Python зависимости
RUN pip3 install --no-cache-dir -r /app/api/requirements.txt

# Копируем весь проект
COPY . /app/

# Устанавливаем CT-CLIP как пакет
RUN cd /app/CT_CLIP && pip3 install -e .

# Устанавливаем transformer_maskgit как пакет
RUN cd /app/transformer_maskgit && pip3 install -e .

# Устанавливаем fvlm как пакет
RUN pip3 install -e fvlm

# Создаем директорию для моделей
RUN mkdir -p /app/models

# Копируем модели из best_models
COPY best_models/supervised_model.pt /app/models/
COPY best_models/CT_VocabFine_v2.pt /app/models/
COPY lightgbm_results/final_lightgbm_model.pkl /app/models/

COPY fvlm/weights /app/models/fvlm/

# Порт для API
EXPOSE 8000

# Переменные окружения
ENV SUPERVISED_MODEL_PATH=/app/models/supervised_model.pt
ENV CTCLIP_MODEL_PATH=/app/models/CT_VocabFine_v2.pt
ENV LIGHTGBM_MODEL_PATH=/app/models/final_lightgbm_model.pkl
ENV OPTIMAL_THRESHOLD=0.5

ENV FVLM_MODEL_PATH=/app/models/fvlm/fvlm_model.pth
ENV FVLM_MAE_WEIGHTS_PATH=/app/models/fvlm/mae_pretrain_vit_base.pth
ENV FVLM_BERT_PATH=/app/models/fvlm/BiomedVLP-CXR-BERT-specialized

ENV DEVICE=cuda
ENV PYTHONPATH=/app:$PYTHONPATH

# Запуск API
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
