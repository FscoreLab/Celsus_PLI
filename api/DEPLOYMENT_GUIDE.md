# Руководство по развертыванию CT-CLIP Inference API

Полное руководство по установке и настройке CT-CLIP Inference API на production сервере.

---

## 📋 Содержание

1. [Предварительные требования](#предварительные-требования)
2. [Установка зависимостей](#установка-зависимостей)
3. [Подготовка моделей](#подготовка-моделей)
4. [Настройка Docker](#настройка-docker)
5. [Запуск с Docker](#запуск-с-docker)
6. [Настройка GPU](#настройка-gpu)
7. [Масштабирование](#масштабирование)
8. [Мониторинг и логирование](#мониторинг-и-логирование)
9. [Резервное копирование](#резервное-копирование)
10. [Безопасность](#безопасность)

---

## 🔧 Предварительные требования

### Аппаратные требования

#### Рекомендуемые (GPU инференс)
- CPU: 16+ ядер
- RAM: 32 GB
- GPU: NVIDIA GPU с 8+ GB VRAM (RTX 3070, A4000, V100, A10, A100)
- Диск: 100 GB NVMe SSD
- Сеть: 1 Gbit/s

### Программное обеспечение

- Docker >= 20.10
- Docker Compose >= 1.29
- Git
- (Для GPU) NVIDIA Driver >= 525.x
- (Для GPU) NVIDIA Container Toolkit

---

## 📦 Установка зависимостей

### Обновление системы

```bash
sudo apt update && sudo apt upgrade -y
```

### Установка Docker

```bash
# Установка зависимостей
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
lspci | grep -i nvidia

sudo apt install -y nvidia-driver-525

sudo reboot

nvidia-smi
```

### Установка NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo systemctl restart docker

# Проверка
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 📥 Подготовка моделей

### Структура директории моделей

```bash
sudo mkdir -p /app/models
sudo chown -R $USER:$USER /app/models
```

### Скачивание моделей

```
/app/models/
├── CT_CLIP_Supervised_finetune_Zheka.pt   # ~800 MB
├── CT_VocabFine_v2.pt                      # ~400 MB
└── lightgbm_model.pkl                      # ~5 MB
```

```bash
cd /app/models
# Скачайте модели и поместите в эту директорию
ls -lh /app/models/
```

---

## 🐳 Настройка Docker

### Клонирование репозитория

```bash
cd /opt
sudo git clone <REPOSITORY_URL> ct-clip
sudo chown -R $USER:$USER ct-clip
cd ct-clip/api
```

### Настройка docker-compose.yml

Отредактируйте `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ct-clip-api:
    build:
      context: ..
      dockerfile: api/Dockerfile
    container_name: ct-clip-inference-api
    restart: unless-stopped
    
    ports:
      - "8000:8000"
    
    volumes:
      - /app/models:/app/models:ro
      - ./logs:/app/logs
      - /tmp/ct-clip:/tmp:rw
    
    environment:
      - SUPERVISED_MODEL_PATH=/app/models/CT_CLIP_Supervised_finetune_Zheka.pt
      - CTCLIP_MODEL_PATH=/app/models/CT_VocabFine_v2.pt
      - LIGHTGBM_MODEL_PATH=/app/models/lightgbm_model.pkl
      - OPTIMAL_THRESHOLD=0.4
      - DEVICE=cuda
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### Сборка образа

```bash
cd /opt/ct-clip/api
docker-compose build
docker images | grep ct-clip
```

---

## 🚀 Запуск с Docker

### Первый запуск

```bash
cd /opt/ct-clip/api

# Запуск в foreground (для проверки)
docker-compose up

# Если OK, запустить в background
docker-compose down
docker-compose up -d
```

### Проверка работоспособности

```bash
docker-compose ps
docker-compose logs -f

curl http://localhost:8000/health
curl http://localhost:8000/
```

### Тестирование инференса

```bash
zip -r test.zip /path/to/dicom/files/

curl -X POST http://localhost:8000/predict \
  -F "file=@test.zip" \
  -o result.json

cat result.json | jq .
```

---

## 📈 Масштабирование

### Горизонтальное масштабирование

Для обработки большого количества запросов используйте nginx + несколько инстансов API.

#### nginx конфигурация

```nginx
upstream ct_clip_api {
    least_conn;
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 500M;
    
    location / {
        proxy_pass http://ct_clip_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

---

## 📊 Мониторинг и логирование

### Просмотр логов

```bash
docker-compose logs -f
docker-compose logs --tail=100
docker-compose logs -f -t
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

### Мониторинг ресурсов

```bash
docker stats ct-clip-inference-api

nvidia-smi -l 1

watch -n 1 nvidia-smi
```

---

## 💾 Резервное копирование

### Пример скрипта резервного копирования

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/ct-clip"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/$DATE

cp -r /app/models $BACKUP_DIR/$DATE/
cp /opt/ct-clip/api/docker-compose.yml $BACKUP_DIR/$DATE/

cd $BACKUP_DIR
tar -czf ct-clip-backup-$DATE.tar.gz $DATE/
rm -rf $DATE/

find $BACKUP_DIR -name "ct-clip-backup-*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/ct-clip-backup-$DATE.tar.gz"
```

### Настройка автоматического резервного копирования

```bash
chmod +x /opt/ct-clip/backup.sh

crontab -e
# Добавить: 0 2 * * * /opt/ct-clip/backup.sh >> /var/log/ct-clip-backup.log 2>&1
```

---

## 🔄 Обновление системы

### Обновление кода

```bash
cd /opt/ct-clip
git stash
git pull origin main
git stash pop
```

### Пересборка Docker образа

```bash
cd /opt/ct-clip/api
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Обновление моделей

```bash
docker-compose down
cp new_model.pt /app/models/CT_VocabFine_v2.pt
docker-compose up -d
```

---

## 🔒 Безопасность

### Файрвол

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # API (если нужен внешний доступ)
sudo ufw enable
sudo ufw status
```

### HTTPS/SSL

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
sudo systemctl enable certbot.timer
```

### Ограничение доступа в nginx

```nginx
location / {
    allow 192.168.1.0/24;
    deny all;
    proxy_pass http://ct_clip_api;
}

# Basic Auth
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://ct_clip_api;
}
```

---

## 🧪 Тестирование после развертывания

### Функциональное тестирование

```bash
cd /opt/ct-clip/api
python test_client.py --url http://localhost:8000 --file test.zip
```

### Проверка GPU утилизации

```bash
nvidia-smi dmon -s u

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1
```

---

## 📞 Диагностика проблем

```bash
docker-compose ps
docker-compose logs --tail=50

docker stats

nvidia-smi

sudo netstat -tulpn | grep 8000

df -h
```

---

## 📝 Чеклист развертывания

- [ ] Установлен Docker и Docker Compose
- [ ] Установлен NVIDIA Driver и Container Toolkit (для GPU)
- [ ] Скачаны и размещены файлы моделей
- [ ] Склонирован репозиторий
- [ ] Настроен docker-compose.yml
- [ ] Собран Docker образ
- [ ] Запущен сервис
- [ ] Проверен health endpoint
- [ ] Выполнен тестовый инференс
- [ ] Настроен мониторинг
- [ ] Настроено резервное копирование
- [ ] Настроен файрвол
- [ ] (Production) Настроен nginx + SSL

---

**Готово!** Ваш CT-CLIP Inference API развернут и готов к использованию.
