FROM continuumio/miniconda3:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_CONDA_CREATE_ENV_CMD="false" \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.7.1

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# copy project (includes main.py, config.yaml, src/ and xgboost_dir)
COPY . /app

# make sure model artifact folder is readable (no-op if it doesn't exist)
RUN chmod -R a+rX /app/xgboost_dir || true

EXPOSE 8000

# start FastAPI app; remove --reload for production images
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
