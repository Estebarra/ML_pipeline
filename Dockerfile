FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.7.1

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN pip install --no-cache-dir "numba>=0.60.0" \
    && pip install --no-cache-dir git+https://github.com/SeldonIO/alibi-detect.git

RUN pip install --no-cache-dir hydra-core hydra-optuna-sweeper

COPY main.py config.yaml ./

COPY src/ ./src/

RUN mkdir -p outputs multirun

ENV MLFLOW_CONDA_CREATE_ENV_CMD="false"

CMD ["conda", "run", "--no-capture-output", "-n", "app", "python", "main.py"]
