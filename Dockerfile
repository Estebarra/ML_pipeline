FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_CONDA_CREATE_ENV_CMD="false" \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.7.1

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN pip install --no-cache-dir "numba>=0.60.0" \
    && pip install --no-cache-dir git+https://github.com/SeldonIO/alibi-detect.git \
    && pip install --no-cache-dir hydra-core hydra-optuna-sweeper \
    && pip install --no-cache-dir "seaborn>=0.13.2"

# copy project (includes main.py, config.yaml, src/ and xgboost_dir)    
COPY . /app

# make sure model artifact folder is readable (no-op if it doesn't exist)
RUN mkdir -p outputs multirun \
    && chmod -R a+rX /app/xgboost_dir || true

EXPOSE 8000

# start FastAPI app; remove --reload for production images
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]