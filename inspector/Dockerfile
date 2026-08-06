# Build from the REPO ROOT so the shared inference core can be copied in:
#   docker build -f inspector/Dockerfile -t zander-inspector .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY inspector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inspector/ .
COPY ml/infer/ ./ml/infer/

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
