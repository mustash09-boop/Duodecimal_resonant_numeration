# -------- BASE IMAGE --------
FROM python:3.10-slim

# -------- SYSTEM SETUP --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------- PYTHON DEPENDENCIES --------
COPY requirements_cloud.txt /workspace/
RUN pip install --no-cache-dir -r requirements_cloud.txt

# -------- PROJECT COPY --------
# Копируем ВЕСЬ проект (важно: структура должна сохраняться)
COPY . /workspace/

# -------- PYTHONPATH --------
ENV PYTHONPATH=/workspace:/workspace/py

# -------- DEFAULT ENTRYPOINT --------
# (будет переопределён в Batch runnable)
CMD ["python", "-m", "music12.blocks.Block005_job_orchestrator.cloud_entrypoint_cli"]