# Руководство пользователя CT-CLIP Inference API

Руководство по использованию CT-CLIP Inference API для анализа КТ-исследований грудной клетки.

---

## 📋 Содержание

1. [Введение](#введение)
2. [Подготовка DICOM данных](#подготовка-dicom-данных)
3. [Использование API](#использование-api)
4. [Интерпретация результатов](#интерпретация-результатов)
5. [Примеры использования](#примеры-использования)
6. [Пакетная обработка](#пакетная-обработка)
7. [Обработка ошибок](#обработка-ошибок)
8. [FAQ](#faq)

---

## 📖 Введение

### Что такое CT-CLIP Inference API?

CT-CLIP Inference API — это система автоматического анализа компьютерных томограмм (КТ) грудной клетки на основе искусственного интеллекта. Система выявляет патологии и аномалии в КТ-исследованиях.

### Архитектура

Система использует ансамбль из пяти моделей машинного обучения:

1. **CT-CLIP Model** — предобученная модель на основе Vision Transformer
2. **Supervised Model** — энкодер CT-CLIP, дообученный на размеченных данных
3. **Anomaly Diffusion Detector** — модель детекции аномалий на основе диффузии
4. **fVLM Model** — модель с контрастивным обучением на анатомических структурах
5. **MedNeXt Model** — supervised-модель с multilabel-классификацией

Предсказания объединяются с помощью двух **LightGBM** классификаторов:
- **Основная модель** — универсальная модель для всех типов патологий
- **Thoracic модель** — специализированная модель для патологий области ОГК

### Важное предупреждение

⚠️ **ВНИМАНИЕ**: Система предназначена **только для диагностической поддержки**. Все результаты должны быть проверены квалифицированным врачом-рентгенологом. Система НЕ предназначена для самостоятельной постановки диагноза.

---

## 📦 Подготовка DICOM данных

### Требования к данным

#### Обязательные требования

1. **Модальность**: CT (Computed Tomography)
2. **Анатомическая область**: Грудная клетка (Chest)
3. **Минимальное количество срезов**: 20 валидных DICOM файлов
4. **Формат**: DICOM (.dcm)

#### Рекомендации

- **Толщина среза**: 1-5 мм
- **Полнота исследования**: Полное исследование от апексов до диафрагмы
- **Качество**: Без артефактов движения

### Подготовка ZIP-архива

```bash
# Linux/Mac
zip -r study_001.zip /path/to/dicom/files/

# Рекурсивно для сложной структуры
zip -r study.zip . -i "*.dcm"
```

### Проверка архива

```bash
# Просмотр содержимого
unzip -l study.zip

# Поиск DICOM файлов
unzip -l study.zip | grep -i "\.dcm"

# Проверка размера
ls -lh study.zip
```

### Типичные ошибки

❌ **Неправильно:**
- Архив содержит несколько исследований
- В архиве только DICOMDIR без файлов
- Файлы не в формате DICOM
- Менее 20 срезов

✅ **Правильно:**
- Одно полное КТ-исследование
- Все DICOM файлы включены
- Минимум 20 срезов

---

## 🔌 Использование API

### Варианты использования

#### Локальное развертывание

```
http://localhost:8000
```

#### Облачное API

```
http://93.187.188.50:7654
```

### API Эндпоинты

#### Health Check

```bash
# Локальный
curl http://localhost:8000/health

# Облачный
curl http://93.187.188.50:7654/health
```

**Ответ:**
```json
{
  "status": "healthy"
}
```

#### Инференс на DICOM архиве

```bash
# Локальный
curl -X POST http://localhost:8000/predict \
  -F "file=@study.zip" \
  -o result.json

# Облачный
curl -X POST http://93.187.188.50:7654/predict \
  -F "file=@study.zip" \
  -o result.json
```

#### Инференс на NIFTI файле

Если у вас уже есть NIFTI файлы (`.nii.gz` или `.nii`), вы можете использовать их напрямую:

```bash
# Локальный
curl -X POST http://localhost:8000/predict_nifti \
  -F "file=@volume.nii.gz" \
  -o result.json

# Облачный
curl -X POST http://93.187.188.50:7654/predict_nifti \
  -F "file=@volume.nii.gz" \
  -o result.json
```

---

## 🐍 Использование через Python

### Базовый пример (DICOM)

```python
import requests

# Выберите URL
api_url = "http://localhost:8000/predict"  # Локальный
# api_url = "http://93.187.188.50:7654/predict"  # Облачный

archive_path = "study.zip"

with open(archive_path, "rb") as f:
    response = requests.post(api_url, files={"file": f})

if response.status_code == 200:
    result = response.json()
    print(f"Probability: {result['probability_of_pathology']:.2%}")
    print(f"Pathology: {result['pathology']}")
    print(f"Top pathologies: {result['top_pathologies']}")
    
    # Thoracic модель (если доступна)
    if result.get('probability_of_pathology_thoracic') is not None:
        print(f"Probability (Thoracic): {result['probability_of_pathology_thoracic']:.2%}")
        print(f"Pathology (Thoracic): {result['pathology_thoracic']}")
else:
    print(f"Error: {response.status_code}")
```

### Базовый пример (NIFTI)

```python
import requests

# Выберите URL
api_url = "http://localhost:8000/predict_nifti"  # Локальный
# api_url = "http://93.187.188.50:7654/predict_nifti"  # Облачный

nifti_path = "volume.nii.gz"

with open(nifti_path, "rb") as f:
    response = requests.post(api_url, files={"file": f})

if response.status_code == 200:
    result = response.json()
    print(f"Probability: {result['probability_of_pathology']:.2%}")
    print(f"Pathology: {result['pathology']}")
    print(f"Top pathologies: {result['top_pathologies']}")
else:
    print(f"Error: {response.status_code}")
```

### С обработкой ошибок

```python
import requests
from pathlib import Path

def analyze_ct_study(archive_path: str, api_url: str = "http://localhost:8000/predict"):
    """Анализ КТ-исследования с обработкой ошибок"""
    
    if not Path(archive_path).exists():
        print(f"File not found: {archive_path}")
        return None
    
    try:
        with open(archive_path, "rb") as f:
            response = requests.post(
                api_url, 
                files={"file": f}, 
                timeout=300  # 5 минут
            )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("processing_status") == "Success":
                return result
            else:
                print(f"Processing failed: {result.get('error', 'Unknown error')}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
    
    except requests.exceptions.Timeout:
        print("Request timeout")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Использование
result = analyze_ct_study("study.zip")
if result:
    print(f"Study UID: {result['study_uid']}")
    print(f"Probability of pathology: {result['probability_of_pathology']:.2%}")
    print(f"Classification: {result['pathology']}")
    print(f"Most dangerous pathology: {result['most_dangerous_pathology_type']}")
```

---

## 📊 Интерпретация результатов

### Структура ответа

```json
{
  "study_uid": "1.2.840.113619.2.xxx",
  "series_uid": "1.2.840.113619.2.yyy",
  "probability_of_pathology": 0.87,
  "pathology": 1,
  "top_pathologies": ["Pneumonia", "Consolidation", "Atelectasis"],
  "probability_of_pathology_thoracic": 0.92,
  "pathology_thoracic": 1,
  "processing_status": "Success",
  "time_of_processing": 45.67
}
```

### Описание полей

- **`study_uid`** — уникальный идентификатор исследования (DICOM StudyInstanceUID)
- **`series_uid`** — уникальный идентификатор серии (DICOM SeriesInstanceUID)
- **`probability_of_pathology`** — вероятность наличия патологии от основной модели (0.0 - 1.0)
- **`pathology`** — классификация основной модели: 0 (норма) или 1 (патология)
- **`top_pathologies`** — список топ-патологий от основной модели (MedNeXt, FVLM, CT-CLIP)
- **`probability_of_pathology_thoracic`** — вероятность наличия патологии от thoracic модели (0.0 - 1.0)
- **`pathology_thoracic`** — классификация thoracic модели: 0 (норма) или 1 (патология)
- **`processing_status`** — статус обработки: "Success" или "Failure"
- **`time_of_processing`** — время обработки в секундах


## 💻 Примеры использования

### Пример 1: Анализ одного исследования

```python
import requests
import json

def analyze_single_study(archive_path, api_url="http://localhost:8000/predict"):
    """Анализ одного КТ-исследования"""
    
    print(f"Analyzing: {archive_path}")
    
    with open(archive_path, "rb") as f:
        response = requests.post(api_url, files={"file": f})
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n=== Results ===")
        print(f"Study UID: {result['study_uid']}")
        print(f"Series UID: {result['series_uid']}")
        print(f"Probability of pathology: {result['probability_of_pathology']:.2%}")
        print(f"Classification: {'Pathology' if result['pathology'] == 1 else 'Normal'}")
        print(f"Most dangerous pathology: {result['most_dangerous_pathology_type']}")
        print(f"Processing time: {result['time_of_processing']:.2f} sec")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

# Использование
result = analyze_single_study("patient_001.zip")
```

### Пример 2: Анализ с сохранением результата

```python
import requests
import json
from datetime import datetime

def analyze_and_save(archive_path, output_dir="results"):
    """Анализ с сохранением результата в JSON"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    api_url = "http://localhost:8000/predict"
    
    with open(archive_path, "rb") as f:
        response = requests.post(api_url, files={"file": f})
    
    if response.status_code == 200:
        result = response.json()
        
        # Добавление метаданных
        result['analysis_timestamp'] = datetime.now().isoformat()
        result['source_file'] = archive_path
        
        # Сохранение результата
        study_uid = result['study_uid'].replace('.', '_')
        output_file = f"{output_dir}/{study_uid}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return result
    
    return None

# Использование
result = analyze_and_save("patient_001.zip")
```

---

## 📦 Пакетная обработка

### Использование встроенного клиента

```bash
python api/batch_inference_client.py \
  --input-dir /path/to/archives/ \
  --output results.xlsx \
  --api-url http://localhost:8000
```

### Собственный скрипт пакетной обработки

```python
import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def batch_analysis(archives_dir, output_file="results.csv", api_url="http://localhost:8000/predict"):
    """Пакетная обработка архивов"""
    
    archives = list(Path(archives_dir).glob("*.zip"))
    print(f"Found {len(archives)} archives")
    
    results = []
    
    for archive in tqdm(archives, desc="Processing"):
        try:
            with open(archive, "rb") as f:
                response = requests.post(
                    api_url, 
                    files={"file": f}, 
                    timeout=300
                )
            
            if response.status_code == 200:
                result = response.json()
                
                row = {
                    "filename": archive.name,
                    "study_uid": result["study_uid"],
                    "series_uid": result["series_uid"],
                    "probability_of_pathology": result["probability_of_pathology"],
                    "pathology": result["pathology"],
                    "most_dangerous_pathology_type": result["most_dangerous_pathology_type"],
                    "processing_time": result["time_of_processing"],
                    "status": "Success"
                }
            else:
                row = {
                    "filename": archive.name,
                    "status": f"HTTP Error {response.status_code}"
                }
            
            results.append(row)
            
        except Exception as e:
            results.append({
                "filename": archive.name,
                "status": f"Exception: {str(e)}"
            })
    
    # Сохранение результатов
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Статистика
    success_count = df[df["status"] == "Success"].shape[0]
    print(f"\n=== Summary ===")
    print(f"Total: {len(archives)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(archives) - success_count}")
    print(f"Results saved to: {output_file}")
    
    return df

# Использование
df = batch_analysis("archives/", "results.csv")
```

---

## ⚠️ Обработка ошибок

### Типы ошибок

#### 1. HTTP Errors

```python
if response.status_code != 200:
    print(f"HTTP Error: {response.status_code}")
    if response.status_code == 500:
        error_data = response.json()
        print(f"Server error: {error_data.get('error', 'Unknown')}")
```

#### 2. Processing Failures

```python
result = response.json()
if result.get("processing_status") == "Failure":
    print(f"Processing failed: {result.get('error', 'Unknown error')}")
```

#### 3. Timeout Errors

```python
try:
    response = requests.post(api_url, files=files, timeout=300)
except requests.exceptions.Timeout:
    print("Request timeout - study may be too large or server is busy")
```

### Пример с retry логикой

```python
import time
import requests

def analyze_with_retry(archive_path, api_url, max_retries=3):
    """Анализ с повторными попытками"""
    
    for attempt in range(max_retries):
        try:
            with open(archive_path, "rb") as f:
                response = requests.post(
                    api_url, 
                    files={"file": f}, 
                    timeout=300
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("processing_status") == "Success":
                    return result
            
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(5)
        
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1}: Timeout")
            if attempt < max_retries - 1:
                time.sleep(10)
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return None

# Использование
result = analyze_with_retry("study.zip", "http://localhost:8000/predict")
```

---

## ❓ FAQ

### Общие вопросы

**Q: Как долго обрабатывается одно исследование?**

A: Обычно 60-120 секунд.

**Q: Можно ли обрабатывать несколько исследований параллельно?**

A: В текущей версии запросы обрабатываются последовательно.

**Q: Какой максимальный размер архива?**

A: Рекомендуется до 500 MB на один архив.

**Q: Поддерживаются ли другие модальности (MRI, X-Ray)?**

A: Нет, система работает только с CT-исследованиями грудной клетки.

### Технические вопросы

**Q: Как система выбирает серию из архива?**

A: Система автоматически выбирает наиболее подходящую серию. Если нужна конкретная серия, создайте отдельный архив только с ней.

**Q: Что делать, если в архиве несколько исследований?**

A: Система обработает только одно исследование. Рекомендуется создавать отдельные архивы для каждого исследования.

**Q: Какое минимальное количество срезов требуется?**

A: Минимум 20 валидных DICOM срезов.

### Клинические вопросы

**Q: Можно ли использовать результаты для постановки диагноза?**

A: Нет, система предназначена только для диагностической поддержки. Окончательный диагноз должен ставить врач.

---