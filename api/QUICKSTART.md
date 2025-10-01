# Быстрый старт CT-CLIP API

## Запуск с Docker

### 1. Подготовка моделей

Убедитесь, что модели находятся в `/app/models/`:
- `CT_CLIP_Supervised_finetune_Zheka.pt`
- `CT_VocabFine_v2.pt`
- `lightgbm_model.pkl`

### 2. Сборка и запуск

```bash
docker-compose -f api/docker-compose.yml build
docker-compose -f api/docker-compose.yml up -d
```

### 3. Проверка

```bash
curl http://localhost:8000/health
# Ожидаемый ответ: {"status":"healthy"}
```

### 4. Просмотр логов

```bash
docker-compose -f api/docker-compose.yml logs -f
```

## Тестирование

### Подготовка архива

```bash
zip -r test_study.zip /path/to/dicom/files/
```

### Отправка на обработку

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_study.zip" \
  -o result.json

cat result.json | jq .
```

## Остановка

```bash
docker-compose -f api/docker-compose.yml down
```

## Troubleshooting

### CUDA out of memory

Запуск на CPU (медленнее):
```bash
DEVICE=cpu docker-compose -f api/docker-compose.yml up -d
```

### Медленная обработка

Первый запрос может занять до 60-90 секунд (загрузка моделей). Последующие запросы будут быстрее (~30-45 секунд).

---

Полная документация в [README.md](README.md)
