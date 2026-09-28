# dashboard (laptop UI) — pure-Python FastAPI service.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8003

# Short graceful-shutdown timeout: open MJPEG relays never end on their own,
# so without it `docker compose stop` would wait for the kill timeout.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--timeout-graceful-shutdown", "3"]
