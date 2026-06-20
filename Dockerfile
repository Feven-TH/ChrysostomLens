FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_LOCAL_EMBEDDINGS=false

COPY requirements.txt requirements-local-embeddings.txt pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then pip install -r requirements-local-embeddings.txt; fi \
    && pip install -e . --no-deps

EXPOSE 8000

CMD ["uvicorn", "chrysostom_lens.server:app", "--host", "0.0.0.0", "--port", "8000"]
