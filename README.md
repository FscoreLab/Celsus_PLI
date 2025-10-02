# CT-CLIP Inference API

## 📋 Описание решения

**CT-CLIP Inference API** — система автоматического анализа компьютерных томограмм (КТ) грудной клетки. Решение предоставляет REST API для обработки DICOM-архивов и выявления патологий и аномалий.

### Назначение

Система разработана для автоматизированной диагностической поддержки врачей-рентгенологов при анализе КТ-исследований. API принимает ZIP-архив с DICOM файлами и возвращает вероятность наличи патологии в исследовании, а также дополнительную информацию.

### Архитектура решения

Система использует ансамбль из трёх моделей машинного обучения:

1. **CT-CLIP Model** — предобученная модель на основе Vision Transformer (https://arxiv.org/abs/2403.17834)
2. **Supervised Model** — энкодер CT-CLIP, дообученный на размеченных данных с патологиями
3. **Anomaly Diffusion Detector** - модель детекции аномалий на основе диффузии

Предсказания моделей объединяются с помощью модели **LightGBM** на основе градиентного бустинга

---

## ✨ Основные возможности

- Автоматическая распаковка и обработка ZIP-архивов с DICOM-исследованиями
- Базовый алгоритм валидации и выбора серии
- Система возвращает вероятность наличия патологии, а также наиболее вероятную патологию 

---

## 💻 Системные требования

### Минимальные требования

- **ОС**: Linux (Ubuntu 20.04+)
- **Docker**: Docker >= 20.10
- **Docker Compose**: >= 1.29
- **Диск**: 50 GB
- **RAM**: 32 GB
- **GPU**: Nvidia, не менее 12GB vRAM
- 
---

## 🚀 Инструкция (локальное развёртывание)

### 1. Скачивание контейнера с моделями
Выполните команду
```bash
docker compose pull 
```

### 2. Запуск

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

## 🚀 Инструкция (облачное развёртывание)

### 1. Проверка

```bash
curl http://93.187.188.50:7654/health
```

**Ожидаемый ответ:**
```json
{
  "status": "healthy"
}
```

### 2. Пример использования

```bash
zip -r test_study.zip /path/to/dicom/files/

curl -X POST "http://93.187.188.50:7654/predict" \
  -F "file=@test_study.zip" \
  -o result.json

cat result.json | jq .
```


---

## 📁 Структура проекта

```
Celsus_PLI/
├── api/
│   ├── __init__.py                  # Python пакет инициализация
│   ├── main.py                      # FastAPI приложение
│   ├── inference_service.py         # Логика инференса моделей
│   ├── requirements.txt             # Python зависимости
│   ├── Dockerfile                   # Docker образ
│   ├── docker-compose.yml           # Docker Compose конфигурация
│   ├── batch_inference_client.py    # Клиент для пакетной обработки
│   ├── prepare_and_run_inference.py # Утилита подготовки DICOM архивов
│   └── run_api.sh                   # Bash скрипт запуска API
│
├── best_models/                     # Предобученные модели
│   ├── CT_VocabFine_v2.pt          # CT-CLIP модель
│   ├── diffusion_checkpoint_step_4000.pt  # Diffusion модель
│   └── supervised_model.pt          # Supervised модель
│
├── CT_CLIP/                         # CT-CLIP пакет
│   └── ct_clip/                     # Основной модуль
│       ├── ct_clip.py               # Основная модель
│       ├── distributed.py           # Распределённое обучение
│       ├── mlm.py                   # Masked Language Modeling
│       ├── tokenizer.py             # Токенизатор
│       └── visual_ssl.py            # Visual SSL
│
├── diffusion_anomaly/               # Diffusion anomaly detection
│   ├── gaussian_diffusion.py        # Gaussian diffusion процесс
│   ├── unet.py                      # U-Net архитектура
│   ├── script_util.py               # Вспомогательные функции
│   └── ...                          # Другие модули diffusion
│
├── transformer_maskgit/             # Transformer MaskGIT
│   └── transformer_maskgit/
│       ├── attention.py             # Attention механизм
│       └── ctvit.py                 # CT Vision Transformer
│
├── scripts/
│   └── universal_ct_inference.py    # Универсальный инференс скрипт
│
├── ct_clip_classifier.py            # CT-CLIP классификатор
├── README.md                        # Этот файл
├── DEPLOYMENT_GUIDE.md              # Руководство по развертыванию
└── USER_GUIDE.md                    # Руководство пользователя
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
  "pathology": 1,
  "most_dangerous_pathology_type": "Pneumonia",
  "processing_status": "Success",
  "time_of_processing": 45.67
}
```

**Описание полей:**
- `study_uid` — уникальный идентификатор исследования (DICOM StudyInstanceUID)
- `series_uid` — уникальный идентификатор серии (DICOM SeriesInstanceUID)
- `probability_of_pathology` — вероятность наличия патологии (0.0 - 1.0)
- `pathology` — классификация: 0 (норма) или 1 (патология)
- `most_dangerous_pathology_type` — конкретная патология, которая наиболее сильно повлияла на решение модели для данного исследования, или `"No specific pathology detected"` если модель не выявила конкретных патологий
- `processing_status` — статус обработки: "Success" или "Failure"
- `time_of_processing` — время обработки в секундах

**Формат ответа (Failure):**
```json
{
  "study_uid": "",
  "series_uid": "",
  "probability_of_pathology": null,
  "pathology": null,
  "most_dangerous_pathology_type": null,
  "processing_status": "Failure",
  "time_of_processing": 12.34,
  "error": "Описание ошибки"
}
```

---

## 📊 Пакетная обработка

```bash
python api/batch_inference_client.py \
  --input-dir /path/to/archives/ \
  --output results.xlsx \
  --api-url http://localhost:8000
```

---