# ML Pipeline - Pipeline de Machine Learning con MLflow

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning utilizando MLflow para la orquestación, con el objetivo de entrenar y evaluar un modelo XGBoost para predecir el rendimiento estudiantil. El pipeline está diseñado siguiendo las mejores prácticas de MLOps, incluyendo versionado de datos, tracking de experimentos, y reproducibilidad.

## Arquitectura del Pipeline

El pipeline está compuesto por los siguientes pasos:

1. **`load_data`**: Carga de datos desde el conjunto de datos fuente
2. **`clean_data`**: Limpieza y preprocesamiento de los datos
3. **`train_test_split_data`**: División de los datos en conjuntos de entrenamiento y prueba
4. **`train_model`**: Entrenamiento del modelo XGBoost con validación cruzada
5. **`test_model`**: Evaluación del modelo en el conjunto de prueba

## Estructura del Proyecto

```
ML_pipeline/
├── main.py                    # Script principal del pipeline
├── config.yaml               # Configuración principal
├── pyproject.toml            # Dependencias del proyecto
├── Dockerfile                # Imagen Docker
├── docker-compose.yml        # Orquestación de contenedores
├── README.md                 # Este archivo
├── data/                     # Datos del proyecto
│   └── raw/
├── src/                      # Código fuente
│   ├── load_data/           # Módulo de carga de datos
│   ├── clean_data/          # Módulo de limpieza de datos
│   ├── train_test_split_data/ # Módulo de división de datos
│   ├── train_model/         # Módulo de entrenamiento
│   └── test_model/          # Módulo de evaluación
├── tests/                   # Pruebas unitarias
│   └──integration/          # Prueba de integración
├── outputs/                 # Resultados de experimentos
└── multirun/               # Resultados de optimización de hiperparámetros
```

## Requisitos del Sistema

- Python 3.12+
- Docker y Docker Compose
- Git
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
```

**Localmente:**
```bash
# Ejecutar solo carga de datos
python main.py main.steps="load_data"

# Ejecutar múltiples pasos
python main.py main.steps="load_data,clean_data,train_test_split_data"

# Ejecutar entrenamiento y evaluación
python main.py main.steps="train_model,test_model"
```

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

El archivo de configuración principal contiene:

- **main**: Configuración general del pipeline
  - `project_name`: Nombre del proyecto en W&B
  - `experiment_name`: Nombre del experimento
  - `steps`: Pasos a ejecutar ("all" o lista separada por comas)

- **data_processing**: Configuración de procesamiento de datos
  - `dataset`: Nombre del archivo de datos
  - `test_size`: Proporción de datos para prueba
  - `random_state`: Semilla para reproducibilidad
  - `stratify_column`: Columna para estratificación

- **modeling**: Configuración del modelo
  - `xgboost`: Hiperparámetros del modelo XGBoost
  - `random_seed`: Semilla para reproducibilidad
  - `stratify_by`: Columna para estratificación

### Variables de Entorno

Crear un archivo `.env` con las siguientes variables (opcional):

```
WANDB_API_KEY=tu_wandb_api_key
MLFLOW_TRACKING_URI=http://localhost:5000
```

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
- Carga el conjunto de datos desde la fuente
- Registra el dataset como artefacto en MLflow
- Valida la integridad de los datos

### 2. Clean Data (`clean_data`)
- Aplica transformaciones de limpieza
- Maneja valores faltantes
- Realiza encoding de variables categóricas
- Guarda los datos limpios como artefacto

### 3. Train Test Split (`train_test_split_data`)
- Divide los datos en conjuntos de entrenamiento y prueba
- Aplica estratificación si se especifica
- Mantiene la reproducibilidad con semillas fijas

### 4. Train Model (`train_model`)
- Entrena el modelo XGBoost
- Realiza validación cruzada
- Registra métricas y parámetros
- Guarda el modelo entrenado

### 5. Test Model (`test_model`)
- Evalúa el modelo en el conjunto de prueba
- Calcula métricas de rendimiento
- Genera reportes de evaluación

## Desarrollo y Contribución

### Estructura de Código

Cada paso del pipeline es un módulo independiente con:
- `MLproject`: Definición del punto de entrada de MLflow
- `conda.yml`: Entorno conda específico
- `run.py`: Script principal del paso

### Agregar Nuevos Pasos

1. Crear directorio en `src/nuevo_paso/`
2. Implementar `MLproject`, `conda.yml`, y `run.py`
3. Agregar el paso a la lista `STEPS` en `main.py`
4. Actualizar la configuración en `config.yaml`

### Testing

```bash
# Ejecutar tests localmente
poetry run pytest

# Con Docker
docker-compose run --rm pipeline pytest
```

## Solución de Problemas

### Problemas Comunes

1. **Error de permisos en Docker:**
   ```bash
   sudo docker-compose run --rm pipeline python main.py
   ```

2. **Problemas con dependencias:**
   ```bash
   docker-compose build --no-cache
   ```

3. **Error de conexión con W&B:**
   - Verificar `WANDB_API_KEY` en `.env`
   - Ejecutar `wandb login` si es necesario

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

### Logs y Debugging

Los logs se guardan en:
- `outputs/`: Resultados de experimentos individuales
- `multirun/`: Resultados de optimización de hiperparámetros

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto y Soporte

Para preguntas, sugerencias o reportar problemas, por favor crear un issue en el repositorio de GitHub o contactar al autor directamente.
