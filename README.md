# ML Pipeline - Pipeline de Machine Learning con MLflow

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning utilizando MLflow para la orquestación y FastAPI para el despliegue, con el objetivo de entrenar y evaluar un modelo XGBoost para predecir el rendimiento estudiantil. El pipeline está diseñado siguiendo las mejores prácticas de MLOps, incluyendo versionado de datos, tracking de experimentos, detección de drift, y reproducibilidad.

## Arquitectura del Pipeline

El pipeline está compuesto por los siguientes pasos:

1. **`load_data`**: Carga de datos desde el conjunto de datos fuente
2. **`clean_data`**: Limpieza y preprocesamiento de los datos
3. **`train_test_split_data`**: División de los datos en conjuntos de entrenamiento y prueba
4. **`train_model`**: Entrenamiento del modelo XGBoost con validación cruzada y feature engineering
5. **`test_model`**: Evaluación del modelo en el conjunto de prueba
6. **`detect_drift`**: Detección de drift en los datos usando métodos estadísticos

## Estructura del Proyecto

```
ML_pipeline/
├── main.py                    # Script principal del pipeline y FastAPI app
├── config.yaml               # Configuración principal con Hydra
├── pyproject.toml            # Dependencias del proyecto (Poetry)
├── poetry.lock               # Lock file de dependencias
├── Dockerfile                # Imagen Docker para despliegue
├── docker-compose.yml        # Orquestación de contenedores
├── README.md                 # Este archivo
├── xg_config.json            # Configuración generada para XGBoost
├── .env                      # Variables de entorno (no versionado)
├── .env.example              # Ejemplo de variables de entorno
├── .dvc/                     # Configuración de DVC para versionado de datos
├── .dvcignore                # Archivos ignorados por DVC
├── cookie-step/              # Template cookiecutter para nuevos pasos
│   ├── cookiecutter.json     # Configuración del template
│   └── {{cookiecutter.step_name}}/  # Template de paso MLflow
├── data/                     # Datos del proyecto
│   └── raw/                  # Datos sin procesar
├── src/                      # Código fuente
│   ├── __init__.py
│   ├── load_data/           # Módulo de carga de datos
│   ├── clean_data/          # Módulo de limpieza de datos
│   ├── eda/                 # Módulo de análisis exploratorio (EDA)
│   ├── train_test_split_data/ # Módulo de división de datos
│   ├── train_model/         # Módulo de entrenamiento con feature engineering
│   │   ├── run.py
│   │   ├── feature_engineering_pipeline.py
│   │   └── train_model_for_test.py
│   ├── test_model/          # Módulo de evaluación del modelo
│   └── detect_drift/        # Módulo de detección de drift
├── tests/                   # Pruebas unitarias
│   ├── __init__.py
│   ├── integration/         # Pruebas de integración
│   ├── test_load_data.py
│   ├── test_clean_data.py
│   ├── test_train_test_split_data.py
│   ├── test_train_model.py
│   └── test_evaluate_model.py
├── outputs/                 # Resultados de experimentos individuales (Hydra)
├── multirun/               # Resultados de optimización de hiperparámetros (Hydra)
└── DOCO/                   # Documentación adicional
```

## Requisitos del Sistema

- Python 3.11+ (el proyecto usa Python 3.12 en Docker)
- Docker y Docker Compose
- Git
- Poetry (gestor de dependencias)
- DVC (opcional, para versionado de datos)
- Acceso a internet para descarga de dependencias

## Instalación y Configuración

### Opción 1: Usando Docker (Recomendado)

1. **Clonar el repositorio:**
```bash
git clone <url-del-repositorio>
cd ML_pipeline
```

2. **Construir la imagen Docker:**
```bash
docker-compose build
```

3. **Crear archivo de variables de entorno (opcional):**
```bash
# Crear archivo .env con configuraciones específicas
cp .env.example .env
# Editar .env con tus configuraciones
```

### Opción 2: Instalación Local

1. **Instalar Poetry:**
```bash
pip install poetry
```

2. **Instalar dependencias:**
```bash
poetry install
```

3. **Activar el entorno virtual:**
```bash
poetry shell
```

## Uso del Pipeline

### Ejecutar Pipeline Completo

**Con Docker:**
```bash
docker-compose run --rm pipeline python main.py
```

**Localmente:**
```bash
python main.py
```

### Ejecutar Pasos Específicos

**Con Docker:**
```bash
# Ejecutar solo carga de datos
docker-compose run --rm pipeline python main.py main.steps="load_data"

# Ejecutar múltiples pasos
docker-compose run --rm pipeline python main.py main.steps="load_data,clean_data,train_test_split_data"

# Ejecutar entrenamiento y evaluación
docker-compose run --rm pipeline python main.py main.steps="train_model,test_model"

# Ejecutar detección de drift
docker-compose run --rm pipeline python main.py main.steps="detect_drift"
```

**Localmente:**
```bash
# Ejecutar solo carga de datos
python main.py main.steps="load_data"

# Ejecutar múltiples pasos
python main.py main.steps="load_data,clean_data,train_test_split_data"

# Ejecutar entrenamiento y evaluación
python main.py main.steps="train_model,test_model"

# Ejecutar detección de drift
python main.py main.steps="detect_drift"
```

### Desplegar API con FastAPI

El proyecto incluye una API REST construida con FastAPI para servir el modelo entrenado.

**Con Docker:**
```bash
# Construir y ejecutar el contenedor
docker-compose build
docker run -d -p 8000:8000 --name ml-pipeline-container ml-pipeline:latest

# O usando docker-compose
docker-compose up -d
```

**Localmente:**
```bash
# Ejecutar el servidor FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints disponibles:**
- `GET /`: Información general y estado del servicio
- `GET /info`: Metadata del servicio y modelo
- `GET /health`: Health check (liveness probe)
- `GET /ready`: Readiness check (verifica si el modelo está cargado)
- `GET /model_files`: Lista archivos del modelo (diagnóstico)
- `GET /model_load_test`: Prueba de carga del modelo
- `POST /model_execution`: Inferencia para un registro individual
- `POST /batch_model_execution`: Inferencia para múltiples registros
- `GET /docs`: Documentación interactiva de la API (Swagger UI)

### Optimización de Hiperparámetros

El proyecto incluye soporte para optimización automática de hiperparámetros usando Optuna:

**Con Docker:**
```bash
docker-compose run --rm pipeline python main.py --multirun
```

**Localmente:**
```bash
python main.py --multirun
```

## Configuración

### Archivo `config.yaml`

El archivo de configuración principal usa Hydra y contiene:

- **main**: Configuración general del pipeline
  - `project_name`: Nombre del proyecto en W&B (default: "my_ml_project")
  - `experiment_name`: Nombre del experimento (default: "experiment")
  - `steps`: Pasos a ejecutar ("all" o lista separada por comas)

- **data_processing**: Configuración de procesamiento de datos
  - `dataset`: Nombre del archivo de datos (default: "student_entry_performance_modified.csv")
  - `test_size`: Proporción de datos para prueba (default: 0.2)
  - `random_state`: Semilla para reproducibilidad (default: 42)
  - `stratify_column`: Columna para estratificación (default: "Performance")

- **modeling**: Configuración del modelo
  - `random_seed`: Semilla para reproducibilidad (default: 42)
  - `stratify_by`: Columna para estratificación (default: "Performance")
  - `xgboost`: Hiperparámetros del modelo XGBoost
    - `n_estimators`: 500
    - `learning_rate`: 0.1
    - `max_depth`: 3
    - `min_child_weight`: 1
    - `subsample`: 0.8
    - `colsample_bytree`: 0.8
    - `gamma`: 0
    - `reg_alpha`: 0
    - `reg_lambda`: 1

- **drift_detection**: Configuración para detección de drift
  - `gmm_components`: Componentes del Gaussian Mixture Model (default: 3)
  - `drift_magnitude`: Magnitud del drift sintético (default: 0.3)
  - `n_samples_synthetic`: Número de muestras sintéticas (default: 1000)
  - `p_val_threshold`: Umbral de p-value para detección (default: 0.01)

- **hydra.sweeper**: Configuración de Optuna para optimización de hiperparámetros
  - `direction`: maximize
  - `n_trials`: 10
  - `study_name`: xgboost_hyperparameter_optimization
  - Rangos de búsqueda para cada hiperparámetro

### Variables de Entorno

Crear un archivo `.env` con las siguientes variables (opcional):

```bash
WANDB_API_KEY=tu_wandb_api_key
WANDB_PROJECT=my_ml_project
MLFLOW_TRACKING_URI=http://localhost:5000
```

Un ejemplo está disponible en [.env.example](.env.example).

## Monitoreo y Tracking

### MLflow

El pipeline utiliza MLflow para:
- Tracking de experimentos
- Versionado de modelos
- Gestión de artefactos
- Reproducibilidad de experimentos

### Weights & Biases (W&B)

Integración con W&B para:
- Visualización de métricas
- Comparación de experimentos
- Gestión de datasets

## Pasos del Pipeline Detallados

### 1. Load Data (`load_data`)
- Carga el conjunto de datos desde `src/load_data/data/`
- Registra el dataset como artefacto en MLflow con el nombre `data.csv`
- Valida la integridad de los datos
- **Artefacto generado**: `data.csv` (tipo: `raw_data`)

### 2. Clean Data (`clean_data`)
- Aplica transformaciones de limpieza a los datos crudos
- Maneja valores faltantes
- Realiza encoding de variables categóricas
- Aplica normalización/estandarización si es necesario
- Guarda los datos limpios como artefacto en MLflow
- **Artefacto de entrada**: `data.csv:latest`
- **Artefacto generado**: `clean_data.csv` (tipo: `cleaned_data`)

### 3. Train Test Split (`train_test_split_data`)
- Divide los datos en conjuntos de entrenamiento/validación y prueba
- Aplica estratificación basada en la columna configurada (default: "Performance")
- Mantiene la reproducibilidad con semillas fijas
- **Artefacto de entrada**: `clean_data.csv:latest`
- **Artefactos generados**:
  - `train_val_data.csv` (datos de entrenamiento/validación)
  - `test_data.csv` (datos de prueba)

### 4. Train Model (`train_model`)
- Aplica feature engineering usando un pipeline personalizado
- Entrena el modelo XGBoost con los hiperparámetros configurados
- Realiza validación cruzada estratificada (StratifiedKFold)
- Registra métricas (accuracy, precision, recall, F1-score) y parámetros en MLflow
- Guarda el modelo entrenado y el pipeline de feature engineering
- Genera archivo `train_results.json` con métricas de validación cruzada
- **Artefacto de entrada**: `train_val_data.csv:latest`
- **Artefacto generado**: `model_export` (modelo entrenado)
- **Ubicación del modelo**: `xgboost_dir/`

### 5. Test Model (`test_model`)
- Carga el modelo entrenado desde MLflow
- Evalúa el modelo en el conjunto de prueba
- Calcula métricas de rendimiento (accuracy, precision, recall, F1-score, ROC-AUC)
- Genera matriz de confusión y reportes de clasificación
- Registra resultados en MLflow
- **Artefactos de entrada**:
  - `model_export:latest`
  - `test_data.csv:latest`

### 6. Detect Drift (`detect_drift`)
- Detecta drift en los datos usando métodos estadísticos avanzados
- Utiliza Gaussian Mixture Models (GMM) para modelar distribuciones
- Genera datos sintéticos con drift controlado para validación
- Aplica pruebas estadísticas (Kolmogorov-Smirnov, Chi-cuadrado)
- Usa Alibi Detect para detección de drift multivariado
- Registra resultados y visualizaciones en MLflow
- **Artefactos de entrada**:
  - `model_export:latest`
  - `test_data.csv:latest`
- **Parámetros configurables**:
  - `gmm_components`: Componentes del GMM
  - `drift_magnitude`: Magnitud del drift sintético
  - `n_samples_synthetic`: Número de muestras para validación
  - `p_val_threshold`: Umbral para detección de drift

## Desarrollo y Contribución

### Estructura de Código

Cada paso del pipeline es un módulo independiente que sigue el estándar MLflow:
- `MLproject`: Definición del punto de entrada de MLflow con parámetros
- `conda.yml`: Especificación del entorno conda con dependencias
- `run.py`: Script principal que implementa la lógica del paso
- `*_for_test.py`: Funciones auxiliares para testing (opcional)

### Agregar Nuevos Pasos

El proyecto incluye un template cookiecutter para facilitar la creación de nuevos pasos:

1. **Usar el template cookiecutter:**
   ```bash
   cd cookie-step
   cookiecutter .
   ```
   Esto creará un nuevo directorio con la estructura básica del paso.

2. **Manual:**
   - Crear directorio en `src/nuevo_paso/`
   - Implementar `MLproject`, `conda.yml`, y `run.py`
   - Agregar el paso a la lista `STEPS` en [main.py](main.py)
   - Actualizar la configuración en [config.yaml](config.yaml) si es necesario
   - Crear tests en `tests/test_nuevo_paso.py`

3. **Verificar integración:**
   ```bash
   # Ejecutar solo el nuevo paso
   python main.py main.steps="nuevo_paso"

   # Ejecutar tests
   pytest tests/test_nuevo_paso.py
   ```

### Testing

```bash
# Ejecutar tests localmente
poetry run pytest

# Con Docker
docker-compose run --rm pipeline pytest
```

## Pruebas Unitarias

Para asegurar la calidad y estabilidad de nuestro código se ha utilizado 'pytest' para realizar pruebas unitarias.

### Configuración del Entorno de Pruebas

1. **Asegurarse de tener un entorno virtual activado**

2. **Instalar pytest**
   ```bash
    pip install pytest
    pip install pytest-cov
    ```

### Ejecución de pruebas

Se pueden ejecutar las pruebas usando los siguientes comandos desde la raíz del proyecto.

#### Ejecutar todas las pruebas

```bash
pytest
```

#### Ejecutar todas las pruebas en modo silencioso

```bash
pytest -q
```

#### Ejecutar pruebas de un módulo específico

```bash
pytest tests/integration/test_pipeline.py
pytest tests/test_load_data.py
pytest tests/test_clean_data.py
pytest tests/test_train_test_split_data.py
pytest tests/test_train_model.py
pytest tests/test_evaluate_model.py
```

#### Ver reporte de cobertura

```bash
pytest --cov=src
```

## Tecnologías Utilizadas

### Core Framework
- **MLflow**: Orquestación del pipeline, tracking de experimentos, y gestión de modelos
- **Hydra**: Gestión de configuración y optimización de hiperparámetros
- **FastAPI**: API REST para servir el modelo entrenado
- **Uvicorn**: Servidor ASGI para FastAPI

### Machine Learning
- **XGBoost**: Algoritmo de gradient boosting para clasificación
- **scikit-learn**: Preprocesamiento, validación cruzada, y métricas
- **Pandas & NumPy**: Manipulación y procesamiento de datos

### MLOps y Tracking
- **Weights & Biases (W&B)**: Visualización de experimentos y datasets
- **DVC**: Versionado de datos (opcional)
- **Alibi Detect**: Detección de drift en datos y modelos

### Optimización
- **Optuna**: Optimización bayesiana de hiperparámetros vía Hydra

### Desarrollo
- **Poetry**: Gestión de dependencias y entornos virtuales
- **pytest**: Framework de testing
- **Docker & Docker Compose**: Containerización y despliegue
- **Black & Ruff**: Formateo y linting de código

### Visualización
- **Seaborn**: Visualizaciones estadísticas (detección de drift)

## Dependencias del Proyecto

Ver [pyproject.toml](pyproject.toml) para la lista completa de dependencias.

**Principales:**
- Python ^3.11
- mlflow ^2.10.0
- wandb ^0.16.3
- hydra-core ^1.3.2
- hydra-optuna-sweeper ^1.2.0
- xgboost ^2.0.3
- scikit-learn ^1.4.0
- pandas ^2.2.0
- fastapi (implícito vía uvicorn)

## Solución de Problemas

### Problemas Comunes

1. **Error de permisos en Docker:**
   ```bash
   sudo docker-compose run --rm pipeline python main.py
   ```

2. **Problemas con dependencias:**
   ```bash
   # Reconstruir sin caché
   docker-compose build --no-cache

   # Reinstalar con Poetry
   poetry install --no-cache
   ```

3. **Error de conexión con W&B:**
   - Verificar `WANDB_API_KEY` en `.env`
   - Ejecutar `wandb login` si es necesario
   - Configurar variables de entorno:
     ```bash
     export WANDB_PROJECT=my_ml_project
     export WANDB_RUN_GROUP=experiment
     ```

4. **Modelo no encontrado en FastAPI:**
   - Verificar que el pipeline se ejecutó completamente
   - Confirmar que existe el directorio `xgboost_dir/`
   - Usar endpoint `/model_files` para diagnóstico
   - Revisar endpoint `/model_load_test` para detalles del error

5. **Error en detección de drift:**
   - Asegurar que `alibi-detect` está instalado:
     ```bash
     pip install git+https://github.com/SeldonIO/alibi-detect.git
     ```
   - Verificar que `numba>=0.60.0` está instalado
   - Revisar configuración en `config.yaml` bajo `drift_detection`

### Logs y Debugging

Los logs se guardan en:
- `outputs/`: Resultados de experimentos individuales (Hydra)
  - Cada ejecución crea un subdirectorio con timestamp
  - Contiene `config.yaml` usado y logs de la ejecución
- `multirun/`: Resultados de optimización de hiperparámetros (Hydra)
  - Organizado por estudio y trials
- `mlruns/`: Tracking de MLflow (si se usa tracking local)
- `wandb/`: Archivos locales de Weights & Biases
- `train_results.json`: Métricas de validación cruzada del último entrenamiento

**Debugging de la API:**
```bash
# Ver logs del contenedor
docker logs ml-pipeline-container

# Modo interactivo
docker run -it --rm ml-pipeline:latest /bin/bash

# Ver archivos del modelo
curl http://localhost:8000/model_files

# Test de carga del modelo
curl http://localhost:8000/model_load_test
```

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Ejemplos de Uso de la API

### Inferencia Individual

```bash
# Enviar un registro para predicción
curl -X POST http://localhost:8000/model_execution \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 18,
    "Gender": "Male",
    "Ethnicity": "Caucasian",
    "ParentalEducation": "Bachelor",
    "StudyTimeWeekly": 10,
    "Absences": 2,
    "Tutoring": "Yes",
    "ParentalSupport": "High",
    "Extracurricular": "Yes",
    "Sports": "Yes",
    "Music": "No",
    "Volunteering": "Yes",
    "GPA": 3.5,
    "GradeClass": "A"
  }'
```

### Inferencia en Lote

```bash
# Enviar múltiples registros
curl -X POST http://localhost:8000/batch_model_execution \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "Age": 18,
        "Gender": "Male",
        "StudyTimeWeekly": 10,
        "GPA": 3.5
      },
      {
        "Age": 19,
        "Gender": "Female",
        "StudyTimeWeekly": 15,
        "GPA": 3.8
      }
    ]
  }'
```

### Health Checks

```bash
# Verificar que el servicio está vivo
curl http://localhost:8000/health

# Verificar que el modelo está listo
curl http://localhost:8000/ready

# Obtener información del servicio
curl http://localhost:8000/info
```

## Autor

**Luis Barranco**
- Email: xiuh.estebarra@gmail.com

## Contacto y Soporte

Para preguntas, sugerencias o reportar problemas, por favor crear un issue en el repositorio de GitHub o contactar al autor directamente.

---

**Última actualización**: Noviembre 2025
