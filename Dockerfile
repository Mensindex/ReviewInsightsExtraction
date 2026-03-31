# 1. Используем официальный образ PyTorch с CUDA 12.1
# Это тяжелый образ, но он гарантирует, что Unsloth/Bitsandbytes заработают сразу.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Метаданные и настройки среды
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# 3. Установка системных зависимостей
# git иногда нужен для установки пакетов напрямую с github (например, unsloth)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 4. Работа с зависимостями
COPY requirements.txt .

# Сначала обновляем pip
# Затем ставим requirements.txt
# ВАЖНО: Явно устанавливаем bitsandbytes, так как он закомментирован в файле,
# но критически нужен для Linux/GPU среды.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install bitsandbytes scipy

# 5. Копируем код приложения и веса моделей
# Структура внутри контейнера будет: /app/app и /app/models
COPY app /app/app
COPY models /app/models

# 6. Открываем порт
EXPOSE 8000

# 7. Заманда запуска
# --host 0.0.0.0 обязателен, чтобы контейнер был доступен извне
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]