FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



COPY backend/ ./backend/
COPY frontend/ ./frontend/

ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    HF_HOME=/tmp/hf

RUN mkdir -p /app/backend/data/chroma /tmp/matplotlib /tmp/cache /tmp/hf && \
    chmod -R 777 /app /tmp

EXPOSE 10000

WORKDIR /app/backend
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
