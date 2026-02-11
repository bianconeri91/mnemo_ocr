# Берём официальный образ Python 3.10 (slim — облегчённый вариант)
# Это ключевой момент: мы фиксируем версию Python,
# чтобы не зависеть от Python 3.13 в Gradio runtime.
FROM python:3.10-slim


# Отключаем создание .pyc файлов (кеш байткода)
# Это уменьшает мусор в контейнере
ENV PYTHONDONTWRITEBYTECODE=1 \

    # Логи выводятся сразу (без буферизации)
    # Нужно, чтобы в HF логах ты видел всё в реальном времени
    PYTHONUNBUFFERED=1 \

    # Заставляем Gradio слушать все интерфейсы контейнера
    # Без этого приложение будет доступно только внутри контейнера
    GRADIO_SERVER_NAME=0.0.0.0 \

    # Порт, который ожидает Hugging Face (должен совпадать с app_port в README)
    GRADIO_SERVER_PORT=7860


# Обновляем список пакетов apt
# Устанавливаем системные библиотеки, которые нужны:
# - gcc / g++ / build-essential — для сборки C-зависимостей
# - libglib2.0-0 / libgl1 — нужны opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*


# Создаём рабочую директорию внутри контейнера
# Все дальнейшие команды будут выполняться в /app
WORKDIR /app


# Копируем только requirements.txt
# Это делается отдельно, чтобы Docker мог кешировать слой
COPY requirements.txt .


# Обновляем pip и устанавливаем зависимости
# --no-cache-dir уменьшает размер образа
RUN pip install --upgrade pip && pip install -r requirements.txt


# Копируем весь проект в контейнер
COPY . .


# Открываем порт 7860 внутри контейнера
# Это декларативная команда — для читаемости
EXPOSE 7860


# Команда, которая будет выполнена при старте контейнера
# В твоём случае запускаем app.py
CMD ["python", "app.py"]