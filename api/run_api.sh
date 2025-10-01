#!/usr/bin/env bash
#
# Скрипт для запуска CT-CLIP + LightGBM API
#

set -e

echo "========================================="
echo "CT-CLIP + LightGBM API Server"
echo "========================================="

# Пути к моделям (настройте под вашу систему)
export SUPERVISED_MODEL_PATH="${SUPERVISED_MODEL_PATH:-/media/crazyfrogspb/Repos/CT-CLIP/best_models/CT_CLIP_Supervised_finetune_Zheka.pt}"
export CTCLIP_MODEL_PATH="${CTCLIP_MODEL_PATH:-/media/crazyfrogspb/Repos/CT-CLIP/best_models/CT_VocabFine_v2.pt}"
export LIGHTGBM_MODEL_PATH="${LIGHTGBM_MODEL_PATH:-/media/crazyfrogspb/Repos/CT-CLIP/lightgbm_results/final_lightgbm_model.pkl}"
export OPTIMAL_THRESHOLD="${OPTIMAL_THRESHOLD:-0.5}"
export DEVICE="${DEVICE:-cuda}"

echo ""
echo "Configuration:"
echo "  SUPERVISED_MODEL_PATH: $SUPERVISED_MODEL_PATH"
echo "  CTCLIP_MODEL_PATH: $CTCLIP_MODEL_PATH"
echo "  LIGHTGBM_MODEL_PATH: $LIGHTGBM_MODEL_PATH"
echo "  OPTIMAL_THRESHOLD: $OPTIMAL_THRESHOLD"
echo "  DEVICE: $DEVICE"
echo ""

# Проверка существования моделей
if [ ! -f "$SUPERVISED_MODEL_PATH" ]; then
    echo "❌ Supervised model not found: $SUPERVISED_MODEL_PATH"
    exit 1
fi

if [ ! -f "$CTCLIP_MODEL_PATH" ]; then
    echo "❌ CT-CLIP model not found: $CTCLIP_MODEL_PATH"
    exit 1
fi

if [ ! -f "$LIGHTGBM_MODEL_PATH" ]; then
    echo "❌ LightGBM model not found: $LIGHTGBM_MODEL_PATH"
    exit 1
fi

echo "✅ All models found"
echo ""

# Активировать виртуальное окружение если есть
if [ -d "/home/crazyfrogspb/.virtualenvs/breastcancer" ]; then
    echo "Activating virtualenv..."
    source /home/crazyfrogspb/.virtualenvs/breastcancer/bin/activate
fi

# Переход в директорию api
cd "$(dirname "$0")"

echo "Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
echo ""

# Запуск сервера
python main.py
