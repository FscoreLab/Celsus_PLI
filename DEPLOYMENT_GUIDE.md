# Руководство по развертыванию CT-CLIP Inference API

Руководство по установке и запуску CT-CLIP Inference API.

---

## 📋 Содержание

1. [Предварительные требования](#предварительные-требования)
2. [Локальное развертывание](#локальное-развертывание)
3. [Облачное развертывание](#облачное-развертывание)
4. [Настройка GPU](#настройка-gpu)
5. [Мониторинг и логирование](#мониторинг-и-логирование)
6. [Обновление системы](#обновление-системы)
7. [Диагностика проблем](#диагностика-проблем)

---

## 🔧 Предварительные требования

### Системные требования

- **ОС**: Linux (Ubuntu 20.04+)
- **Docker**: Docker >= 20.10
- **Docker Compose**: >= 1.29
- **Диск**: 50 GB свободного места
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU с минимум 12GB VRAM

### Установка Docker

```bash
# Установка зависимостей
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Добавление GPG ключа Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Добавление репозитория Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Установка Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Проверка
docker --version
```

### Установка Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

### Добавление пользователя в группу docker

```bash
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```

---

## 🎮 Настройка GPU

### Установка NVIDIA Driver

```bash
# Проверка наличия GPU
lspci | grep -i nvidia

# Установка драйвера
sudo apt install -y nvidia-driver-525

# Перезагрузка
sudo reboot

# Проверка после перезагрузки
nvidia-smi
```

### Установка NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Перезапуск Docker
sudo systemctl restart docker

# Проверка
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Локальное развертывание

### Шаг 1: Скачивание предсобранного образа

```bash
docker compose pull
```

Эта команда загрузит предсобранный Docker образ с моделями и всеми необходимыми зависимостями.

### Шаг 2: Запуск сервиса

```bash
docker-compose up -d
```

### Шаг 3: Проверка работоспособности

```bash
# Проверка статуса контейнера
docker-compose ps

# Проверка health endpoint
curl http://localhost:8000/health
```

**Ожидаемый ответ:**
```json
{
  "status": "healthy"
}
```

### Шаг 4: Тестирование

```bash
# Подготовка тестового архива
zip -r test_study.zip /path/to/dicom/files/

# Отправка на обработку
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_study.zip" \
  -o result.json

# Просмотр результата
cat result.json | jq .
```

---

## ☁️ Облачное развертывание

### Использование облачного API

Система доступна в облаке без необходимости локальной установки:

**URL:** `http://93.187.188.50:7654`

### Проверка доступности

```bash
curl http://93.187.188.50:7654/health
```

**Ожидаемый ответ:**
```json
{
  "status": "healthy"
}
```

### Пример использования

```bash
zip -r test_study.zip /path/to/dicom/files/

curl -X POST "http://93.187.188.50:7654/predict" \
  -F "file=@test_study.zip" \
  -o result.json

cat result.json | jq .
```

---

## 📊 Мониторинг и логирование

### Просмотр логов

```bash
# Все логи в реальном времени
docker-compose logs -f

# Последние 100 строк
docker-compose logs --tail=100

# Логи с временными метками
docker-compose logs -f -t
```

### Мониторинг ресурсов

```bash
# Мониторинг Docker контейнера
docker stats

# Мониторинг GPU
nvidia-smi -l 1

# Альтернативный мониторинг GPU
watch -n 1 nvidia-smi
```

### Настройка ротации логов

Создайте `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "10"
  }
}
```

Перезапустите Docker:
```bash
sudo systemctl restart docker
```

---

## 🔄 Обновление системы

### Обновление Docker образа

```bash
# Остановка сервиса
docker-compose down

# Загрузка новой версии образа
docker compose pull

# Запуск обновленного сервиса
docker-compose up -d
```

### Проверка после обновления

```bash
docker-compose ps
docker-compose logs --tail=50
curl http://localhost:8000/health
```

---

## 🛑 Управление сервисом

### Остановка сервиса

```bash
docker-compose down
```

### Перезапуск сервиса

```bash
docker-compose restart
```

### Просмотр статуса

```bash
docker-compose ps
```

---

## 📞 Диагностика проблем

### Общая диагностика

```bash
# Статус контейнеров
docker-compose ps

# Последние логи
docker-compose logs --tail=50

# Статистика использования ресурсов
docker stats

# Статус GPU
nvidia-smi

# Проверка порта
sudo netstat -tulpn | grep 8000

# Проверка свободного места
df -h
```

### Частые проблемы

#### Сервис не стартует

**Диагностика:**
```bash
docker-compose logs -f
```

**Возможные причины:**
- Порт 8000 занят другим процессом
- Недостаточно памяти GPU
- Проблемы с NVIDIA Container Toolkit

#### Ошибка "CUDA out of memory"

**Решение:**
- Убедитесь, что GPU имеет минимум 12GB VRAM
- Закройте другие процессы, использующие GPU
- Проверьте: `nvidia-smi`

#### Ошибка при обработке DICOM

**Диагностика:**
```bash
# Проверьте содержимое архива
unzip -l archive.zip

# Убедитесь, что файлы в формате DICOM
unzip -l archive.zip | grep -i dcm
```

---


## 📝 Чеклист развертывания

### Локальное развертывание

- [ ] Установлен Docker и Docker Compose
- [ ] Установлен NVIDIA Driver
- [ ] Установлен NVIDIA Container Toolkit
- [ ] Выполнена команда `docker compose pull`
- [ ] Запущен сервис `docker-compose up -d`
- [ ] Проверен health endpoint
- [ ] Выполнен тестовый инференс
- [ ] Настроен мониторинг

### Облачное использование

- [ ] Проверена доступность облачного API
- [ ] Выполнен тестовый запрос
- [ ] Настроены скрипты для пакетной обработки

---
