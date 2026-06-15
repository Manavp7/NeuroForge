# NeuroForge backend — research/simulation only (not a medical device).
FROM python:3.12-slim

WORKDIR /app

# System deps occasionally needed by RDKit wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY neuroforge ./neuroforge
RUN pip install --no-cache-dir -e .

ENV NEUROFORGE_DB=/data/neuroforge.db
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "neuroforge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
