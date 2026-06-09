# receiver (laptop ingestion hub) — pure-Python FastAPI service.
FROM python:3.11-slim

WORKDIR /app

# curl is only here so the compose healthcheck can probe /api/v1/health.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
