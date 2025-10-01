# CT-CLIP Inference API

## 📋 Описание решения

**CT-CLIP Inference API** — система автоматического анализа компьютерных томограмм (КТ) грудной клетки на основе глубокого обучения. Решение предоставляет REST API для обработки DICOM-архивов и выявления более 60 различных патологий.

### Назначение

Система разработана для автоматизированной диагностической поддержки врачей-рентгенологов при анализе КТ-исследований. API принимает ZIP-архив с DICOM файлами и возвращает вероятности наличия различных патологий.

### Архитектура решения

Система использует ансамбль из трёх моделей машинного обучения:

1. **Supervised Model** — специализированная модель на основе CT-CLIP для 16 основных патологий
2. **CT-CLIP Model** — универсальная модель для 50+ патологий на основе Vision Transformer
3. **LightGBM Classifier** — финальный классификатор на основе градиентного бустинга

---

## ✨ Основные возможности

- ✅ **Обработка DICOM архивов**: Автоматическая распаковка и обработка ZIP-архивов с DICOM файлами
- ✅ **Множественные патологии**: Одновременное выявление 60+ различных патологий
- ✅ **Умный выбор серии**: Автоматический выбор оптимальной CT-серии
- ✅ **REST API**: Удобный HTTP-интерфейс для интеграции
- ✅ **Docker**: Полная контейнеризация с поддержкой GPU
- 🚀 **GPU ускорение**: Поддержка NVIDIA CUDA для быстрого инференса
- 📊 **Batch обработка**: Возможность обработки множества исследований

---

## ⚠️ Ограничения системы

### Технические ограничения

1. **Модальность**: Только CT (Computed Tomography)
2. **Анатомическая область**: Оптимизирована для КТ грудной клетки
3. **Минимальное количество срезов**: Требуется минимум 20 валидных DICOM срезов
4. **Формат входных данных**: Только ZIP архивы с DICOM файлами
5. **Размер архива**: Рекомендуется до 500 MB

### Клинические ограничения

1. **Диагностическая поддержка**: Не заменяет врачебную оценку
2. **Требуется экспертная оценка**: Все результаты должны быть проверены специалистом
3. **Обучающая выборка**: Модели обучены на взрослых пациентах

### Производительность

- **Время обработки**: ~30-60 секунд на исследование
- **Память GPU**: ~6-8 GB GPU памяти
- **Последовательная обработка**: Один запрос за раз

---

## 💻 Системные требования

### Минимальные требования

- **ОС**: Linux (Ubuntu 20.04+) или Windows с WSL2
- **Docker**: Docker >= 20.10
- **Docker Compose**: >= 1.29
- **Диск**: 20 GB
- **RAM**: 16 GB

### Рекомендуемые требования

- **ОС**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA GPU с 8+ GB памяти (RTX 3070, A4000 или лучше)
- **CUDA**: 11.8+ (через NVIDIA Container Toolkit)
- **RAM**: 32 GB
- **Диск**: 50 GB

### Для GPU инференса

- NVIDIA GPU с поддержкой CUDA
- NVIDIA Driver >= 525.x
- NVIDIA Container Toolkit

---

## 🚀 Быстрый старт

### 1. Подготовка моделей

Поместите модели в директорию `/app/models/`:

```
/app/models/
├── CT_CLIP_Supervised_finetune_Zheka.pt
├── CT_VocabFine_v2.pt
└── lightgbm_model.pkl
```

### 2. Сборка и запуск

```bash
docker-compose build
docker-compose up -d
```

### 3. Проверка

```bash
curl http://localhost:8000/health
```

**Ожидаемый ответ:**
```json
{
  "status": "healthy"
}
```

### 4. Пример использования

```bash
zip -r test_study.zip /path/to/dicom/files/

curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_study.zip" \
  -o result.json

cat result.json | jq .
```

Подробная инструкция: [QUICKSTART.md](QUICKSTART.md)

---

## 📁 Структура проекта

```
api/
├── main.py                          # FastAPI приложение
├── inference_service.py             # Логика инференса моделей
├── requirements.txt                 # Python зависимости
├── Dockerfile                       # Docker образ
├── docker-compose.yml               # Docker Compose конфигурация
├── batch_inference_client.py        # Клиент для пакетной обработки
├── prepare_and_run_inference.py     # Утилита подготовки DICOM архивов
├── README.md                        # Этот файл
├── DEPLOYMENT_GUIDE.md              # Руководство по развертыванию
├── USER_GUIDE.md                    # Руководство пользователя
└── QUICKSTART.md                    # Быстрый старт
```

---

## 🔌 API Эндпоинты

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "healthy"
}
```

### `POST /predict` — Инференс на DICOM архиве

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/dicom_archive.zip"
```

**Формат ответа (Success):**
```json
{
  "study_uid": "1.2.840.113619.2.xxx",
  "series_uid": "1.2.840.113619.2.yyy",
  "probability_of_pathology": 0.87,
  "pathology": "Патология",
  "most_dangerous_pathology_type": "Pneumonia",
  "processing_status": "Success",
  "time_of_processing": 45.67,
  "supervised_probabilities": { ... },
  "ctclip_probabilities": { ... },
  "top_shap_features": { ... }
}
```

**Формат ответа (Failure):**
```json
{
  "study_uid": "",
  "series_uid": "",
  "probability_of_pathology": 0.0,
  "pathology": "Unknown",
  "most_dangerous_pathology_type": "Unknown",
  "processing_status": "Failure",
  "time_of_processing": 12.34,
  "error": "Описание ошибки"
}
```

---

## 📊 Примеры использования

### Python клиент

```python
import requests

api_url = "http://localhost:8000/predict"
dicom_archive = "study.zip"

with open(dicom_archive, "rb") as f:
    response = requests.post(api_url, files={"file": f})

if response.status_code == 200:
    result = response.json()
    print(f"Вероятность патологии: {result['probability_of_pathology']:.2%}")
    print(f"Диагноз: {result['pathology']}")
```

### Пакетная обработка

```bash
python api/batch_inference_client.py \
  --input-dir /path/to/archives/ \
  --output results.xlsx \
  --api-url http://localhost:8000
```

---

## 🔧 Настройка

### Переменные окружения

Настройте в `docker-compose.yml`:

```yaml
environment:
  - SUPERVISED_MODEL_PATH=/app/models/CT_CLIP_Supervised_finetune_Zheka.pt
  - CTCLIP_MODEL_PATH=/app/models/CT_VocabFine_v2.pt
  - LIGHTGBM_MODEL_PATH=/app/models/lightgbm_model.pkl
  - OPTIMAL_THRESHOLD=0.4
  - DEVICE=cuda  # или cpu
```

### Изменение портов

В `docker-compose.yml`:

```yaml
ports:
  - "8000:8000"  # Изменить первый порт
```

---

## 🩺 Поддерживаемые патологии

### Supervised модель (16 классов)

ribs_fracture, pleural_effusion, aorta_pathology, pulmonary_trunk_pathology, cancer, atelectasis, covid, infiltrate, emphysema, paracardial_fat_pathology, fibrosis, airiness_decrease, pneumothorax, coronary_calcium, osteo_fracture

### CT-CLIP модель (50+ классов)

Pneumonia, Tuberculosis, Bronchiectasis, Emphysema, Fibrosis, Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary nodule, Thymoma, Lymphoma, Aortic dissection, Pulmonary embolism, и многое другое...

---

## 🛠️ Troubleshooting

### Ошибка "CUDA out of memory"

Используйте CPU инференс:
```bash
DEVICE=cpu docker-compose up -d
```

### Ошибка "No DICOM series found"

1. Проверьте содержимое архива: `unzip -l archive.zip | grep -i dcm`
2. Убедитесь, что файлы в формате DICOM
3. Проверьте логи: `docker-compose logs -f`

### Ошибка "Недостаточно валидных срезов"

Убедитесь, что архив содержит минимум 20 валидных DICOM файлов.

### Сервис не стартует

```bash
docker-compose logs -f
docker-compose down && docker-compose up -d
```

---

## 📚 Дополнительные ресурсы

- **[QUICKSTART.md](QUICKSTART.md)** — Быстрый старт
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** — Полное руководство по развертыванию
- **[USER_GUIDE.md](USER_GUIDE.md)** — Руководство пользователя с примерами

---

## 📊 Логирование

```bash
# Просмотр логов
docker-compose logs -f

# Последние 100 строк
docker-compose logs --tail=100
```

---

## 🛑 Остановка сервиса

```bash
docker-compose down

# С удалением volumes
docker-compose down -v
```

---

**Версия:** 2.0.0  
**Дата обновления:** 2025-10-01
