# Руководство пользователя CT-CLIP Inference API

Руководство по использованию CT-CLIP Inference API для анализа КТ-исследований грудной клетки.

---

## 📋 Содержание

1. [Введение](#введение)
2. [Подготовка DICOM данных](#подготовка-dicom-данных)
3. [Использование API](#использование-api)
4. [Интерпретация результатов](#интерпретация-результатов)
5. [Примеры использования](#примеры-использования)
6. [Обработка ошибок](#обработка-ошибок)
7. [FAQ](#faq)

---

## 📖 Введение

### Что такое CT-CLIP Inference API?

CT-CLIP Inference API — это система искусственного интеллекта для автоматического анализа компьютерных томограмм (КТ) грудной клетки. Система выявляет более 60 различных патологий и предоставляет результаты в виде вероятностей.

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

- **Реконструкция**: Лёгочная (LUNG) или стандартная (STANDARD)
- **Толщина среза**: 1-5 мм
- **Без контраста**: Система оптимизирована для нативных исследований
- **Полнота исследования**: Полное исследование от апексов до диафрагмы

### Подготовка ZIP-архива

```bash
# Linux/Mac
zip -r study_001.zip /path/to/dicom/files/

# Или рекурсивно для сложной структуры
zip -r study.zip . -i "*.dcm"
```

### Проверка архива

```bash
unzip -l study.zip
unzip -l study.zip | grep -i "\.dcm"
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

### Базовый URL

```
http://localhost:8000
```

### Доступные эндпоинты

#### Health Check

```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "healthy"
}
```

#### Инференс

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/study.zip" \
  -o result.json
```

### Использование через Python

#### Базовый пример

```python
import requests

api_url = "http://localhost:8000/predict"
archive_path = "study.zip"

with open(archive_path, "rb") as f:
    response = requests.post(api_url, files={"file": f})

if response.status_code == 200:
    result = response.json()
    print(f"Probability: {result['probability_of_pathology']:.2%}")
else:
    print(f"Error: {response.status_code}")
```

#### С обработкой ошибок

```python
import requests
from pathlib import Path

def analyze_ct_study(archive_path: str, api_url: str = "http://localhost:8000/predict"):
    if not Path(archive_path).exists():
        return None
    
    try:
        with open(archive_path, "rb") as f:
            response = requests.post(api_url, files={"file": f}, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("processing_status") == "Success":
                return result
    
    except requests.exceptions.Timeout:
        print("Timeout error")
    except Exception as e:
        print(f"Error: {e}")
    
    return None

result = analyze_ct_study("study.zip")
if result:
    print(f"Pathology: {result['pathology']}")
```

---

## 📊 Интерпретация результатов

### Структура ответа

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

### Описание полей

- **`probability_of_pathology`** — вероятность наличия патологии (0.0 - 1.0)
- **`pathology`** — классификация: "Норма" или "Патология"
- **`most_dangerous_pathology_type`** — наиболее вероятная патология
- **`supervised_probabilities`** — вероятности от Supervised модели (16 патологий)
- **`ctclip_probabilities`** — вероятности от CT-CLIP модели (50+ патологий)
- **`top_shap_features`** — важность признаков (SHAP values)

### Интерпретация вероятностей

#### probability_of_pathology

| Вероятность | Интерпретация |
|-------------|---------------|
| < 0.4 | Скорее всего норма |
| 0.4 - 0.6 | Пограничное состояние |
| 0.6 - 0.8 | Вероятна патология |
| > 0.8 | Высокая вероятность патологии |

#### Индивидуальные патологии

| Вероятность | Интерпретация |
|-------------|---------------|
| < 0.3 | Маловероятно |
| 0.3 - 0.5 | Возможно присутствует |
| 0.5 - 0.7 | Вероятно присутствует |
| > 0.7 | Высокая вероятность |

### Примеры интерпретации

#### Пример 1: Норма

```json
{
  "probability_of_pathology": 0.12,
  "pathology": "Норма",
  "most_dangerous_pathology_type": "No significant pathology"
}
```

**Интерпретация**: Все вероятности низкие, значимых патологий не выявлено.

#### Пример 2: Пневмония

```json
{
  "probability_of_pathology": 0.92,
  "pathology": "Патология",
  "most_dangerous_pathology_type": "Pneumonia",
  "ctclip_probabilities": {
    "ctclip_Pneumonia": 0.93,
    "ctclip_Consolidation": 0.78,
    "ctclip_Ground-glass opacity": 0.65
  }
}
```

**Интерпретация**: Высокая вероятность пневмонии с признаками инфильтрации и консолидации. Требуется клиническая корреляция.

---

## 💻 Примеры использования

### Пример 1: Анализ одного исследования

```python
import requests

def analyze_single_study(archive_path):
    api_url = "http://localhost:8000/predict"
    
    with open(archive_path, "rb") as f:
        response = requests.post(api_url, files={"file": f})
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"Study: {result['study_uid']}")
        print(f"Probability: {result['probability_of_pathology']:.2%}")
        print(f"Classification: {result['pathology']}")
        print(f"Most dangerous: {result['most_dangerous_pathology_type']}")
        
        # Топ-10 патологий
        all_probs = {
            **result['supervised_probabilities'],
            **result['ctclip_probabilities']
        }
        top10 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 Pathologies:")
        for i, (name, prob) in enumerate(top10, 1):
            clean_name = name.replace("supervised_", "").replace("ctclip_", "")
            print(f"  {i:2d}. {clean_name:40s} {prob:.3f}")
        
        return result

result = analyze_single_study("patient_001.zip")
```

### Пример 2: Пакетная обработка

```python
import requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def batch_analysis(archives_dir, output_file="results.csv"):
    api_url = "http://localhost:8000/predict"
    archives = list(Path(archives_dir).glob("*.zip"))
    
    results = []
    
    for archive in tqdm(archives):
        try:
            with open(archive, "rb") as f:
                response = requests.post(api_url, files={"file": f}, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                row = {
                    "filename": archive.name,
                    "study_uid": result["study_uid"],
                    "probability_of_pathology": result["probability_of_pathology"],
                    "pathology": result["pathology"],
                    "most_dangerous": result["most_dangerous_pathology_type"],
                    "status": "Success"
                }
            else:
                row = {"filename": archive.name, "status": f"Error: {response.status_code}"}
            
            results.append(row)
            
        except Exception as e:
            results.append({"filename": archive.name, "status": f"Exception: {e}"})
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    success_count = df[df["status"] == "Success"].shape[0]
    print(f"Successfully processed: {success_count}/{len(archives)}")
    
    return df

df = batch_analysis("archives/", "results.csv")
```

---

## ⚠️ Обработка ошибок

### Типы ошибок

#### HTTP 400: Bad Request

**Причина**: Неправильный формат запроса или файла

```python
if not filepath.endswith('.zip'):
    print("Error: File must be a ZIP archive")
```

#### HTTP 500: Internal Server Error

**Причина**: Ошибка при обработке

```python
if response.status_code == 500:
    result = response.json()
    error_msg = result.get("error", "Unknown error")
    print(f"Processing error: {error_msg}")
```

#### Timeout

**Причина**: Обработка занимает слишком много времени

```python
response = requests.post(api_url, files=files, timeout=600)  # 10 минут
```

### Пример с retry

```python
import time

def robust_analysis(archive_path, max_retries=3):
    api_url = "http://localhost:8000/predict"
    
    for attempt in range(max_retries):
        try:
            with open(archive_path, "rb") as f:
                response = requests.post(api_url, files={"file": f}, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("processing_status") == "Success":
                    return result
            
            time.sleep(5)
        
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt+1}: Timeout")
            continue
        except Exception as e:
            return {"error": str(e)}
    
    return {"error": f"Failed after {max_retries} attempts"}

result = robust_analysis("study.zip")
```

---

## ❓ FAQ

### Общие вопросы

**Q: Как долго обрабатывается одно исследование?**

A: 30-60 секунд с GPU, 5-10 минут с CPU.

**Q: Можно ли обрабатывать параллельно?**

A: В текущей версии запросы обрабатываются последовательно.

**Q: Какой максимальный размер архива?**

A: Рекомендуется до 500 MB.

### Технические вопросы

**Q: Почему система выбрала не ту серию?**

A: Система выбирает серию по приоритету:
1. Лёгочная реконструкция (LUNG kernel)
2. Стандартная реконструкция
3. Серия с наибольшим количеством срезов

**Q: Что делать, если в архиве несколько серий?**

A: Система автоматически выберет лучшую. Если нужна конкретная серия, создайте отдельный архив.

**Q: Поддерживаются ли контрастные исследования?**

A: Да, но система оптимизирована для нативных КТ.

### Клинические вопросы

**Q: Можно ли использовать для диагностики COVID-19?**

A: Система может выявить признаки COVID-пневмонии, но окончательный диагноз требует клинической корреляции и ПЦР-теста.

**Q: Насколько точна система?**

A: Точность зависит от конкретной патологии. Система показывает хорошую чувствительность и специфичность, но всегда требуется экспертная оценка.

---

## 💡 Лучшие практики

### Подготовка данных

1. Используйте полные исследования от апексов до диафрагмы
2. Проверяйте качество DICOM файлов
3. Один архив = одно исследование
4. Оптимальная толщина среза: 1-5 мм

### Использование API

1. Устанавливайте таймауты (300+ секунд)
2. Обрабатывайте ошибки — всегда проверяйте `processing_status`
3. Логируйте результаты для аудита
4. Не перегружайте API

### Интерпретация результатов

1. Смотрите на общую картину — учитывайте все вероятности
2. Учитывайте клинический контекст
3. Проверяйте сомнительные случаи
4. Документируйте решения

---

## 📞 Поддержка

Если у вас возникли вопросы:

1. Проверьте [Troubleshooting](README.md#troubleshooting) в README
2. Посмотрите логи: `docker-compose logs -f`
3. Обратитесь к команде разработчиков

---

**Удачи в использовании CT-CLIP Inference API!**
