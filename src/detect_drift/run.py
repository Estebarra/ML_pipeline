#!/usr/bin/env python
import os
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import wandb
import mlflow
from sklearn.mixture import GaussianMixture
from alibi_detect.cd import KSDrift
from scipy.sparse import issparse

# Configuración del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def go(args):
    """
    Ejecuta la detección de Covariate Shift (features) y Prediction Shift (salidas).
    """
    run = wandb.init(job_type="detect_drift")
    run.config.update(args)
    logger.info("*" * 50)
    logger.info("Iniciando proceso de Detección de Drift")
    logger.info("*" * 50)

    # --- 1. Cargar Modelo y Datos ---
    logger.info("Descargando artefactos: Modelo y Datos de Referencia")
    model_local_path = run.use_artifact(args.model_artifact).download()
    reference_dataset_path = run.use_artifact(args.reference_dataset).file()

    logger.info("Cargando modelo MLflow y datos de referencia")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    X_ref_df = pd.read_csv(reference_dataset_path)
    if "Performance" in X_ref_df.columns:
        X_ref_df = X_ref_df.drop("Performance", axis=1)

    # --- 2. Preparar Datos de Referencia ---
    logger.info("Generando predicciones de referencia (P(y_hat))")
    preds_ref_proba = sk_pipe.predict_proba(X_ref_df)
    preds_ref_dist = preds_ref_proba[:, 1] 

    logger.info("Ajustando preprocesador y entrenando GMM")
    preprocessor = sk_pipe.named_steps['preprocesador']
    X_ref_transformed = preprocessor.transform(X_ref_df) 
    if issparse(X_ref_transformed):
        X_ref_transformed = X_ref_transformed.toarray()

    # --- 3. Generar Datos Sintéticos (Drift) ---
    gmm_base = GaussianMixture(n_components=args.gmm_components, random_state=args.random_seed, covariance_type='diag')
    gmm_base.fit(X_ref_transformed)

    logger.info(f"Simulando drift con magnitud: {args.drift_magnitude}")
    gmm_drift = gmm_base
    n_features_to_drift = min(5, X_ref_transformed.shape[1])
    gmm_drift.means_[:, :n_features_to_drift] += args.drift_magnitude

    logger.info(f"Generando {args.n_samples_synthetic} muestras sintéticas")
    X_drift_sintetico_transformed = gmm_drift.sample(args.n_samples_synthetic)[0] 

    logger.info("Generando predicciones sobre datos sintéticos")
    model_only = sk_pipe.named_steps['modelo']
    preds_drift_proba = model_only.predict_proba(X_drift_sintetico_transformed)
    preds_drift_dist = preds_drift_proba[:, 1]

    try:
        logger.info("Registrando tablas de datos para gráficos en W&B")

        # --- GRÁFICO 1: SCATTER PLOT DE FEATURES ---
        FEATURE_INDEX_X = 0 
        FEATURE_INDEX_Y = 1

        logger.info(f"Registrando tabla para Scatter Plot (Features {FEATURE_INDEX_X} vs {FEATURE_INDEX_Y})")
        
        scatter_data = []
        # Extrae datos de referencia
        ref_x = X_ref_transformed[:, FEATURE_INDEX_X]
        ref_y = X_ref_transformed[:, FEATURE_INDEX_Y]
        for x, y in zip(ref_x, ref_y):
            scatter_data.append(["Referencia", x, y])

        # Extrae datos con drift
        drift_x = X_drift_sintetico_transformed[:, FEATURE_INDEX_X]
        drift_y = X_drift_sintetico_transformed[:, FEATURE_INDEX_Y]
        for x, y in zip(drift_x, drift_y):
            scatter_data.append(["Drift", x, y])
            
        # Crea y registra la tabla
        scatter_table = wandb.Table(
            data=scatter_data, 
            columns=["Fuente", f"Feature_{FEATURE_INDEX_X}", f"Feature_{FEATURE_INDEX_Y}"]
        )
        wandb.log({"scatter_plot_features": scatter_table})

        # --- GRÁFICO 2: PREDICCIONES (El que ya tenías) ---
        logger.info("Registrando tabla para Histograma de Predicciones")
        pred_data = []
        for pred in preds_ref_dist:
            pred_data.append(["Referencia", pred])
        for pred in preds_drift_dist:
            pred_data.append(["Drift", pred])
        
        pred_table = wandb.Table(data=pred_data, columns=["Fuente", "Probabilidad"])
        wandb.log({"tabla_comparativa_predicciones": pred_table})

    except Exception as e:
        logger.warning(f"No se pudieron registrar las tablas para gráficos en W&B: {e}")

    # --- 4. Ejecutar Detección de Drift ---
    logger.info("Ejecutando Detección de Covariate Shift (Features)")
    cd_features = KSDrift(X_ref_transformed, p_val=args.p_val_threshold)
    resultado_features = cd_features.predict(X_drift_sintetico_transformed)
    
    feature_drift_detected = bool(resultado_features['data']['is_drift'])
    feature_p_valor_array = resultado_features['data']['p_val']
    feature_p_valor_minimo = feature_p_valor_array.min() 

    if feature_drift_detected:
        logger.warning(f"  ¡¡¡ DRIFT DE FEATURES DETECTADO !!! (p-value mínimo: {feature_p_valor_minimo:.6f})")
    else:
        logger.info(f"  No se detectó drift de features significativo (p-value mínimo: {feature_p_valor_minimo:.6f})")

    logger.info("Ejecutando Detección de Prediction Shift (Salidas)")
    cd_preds = KSDrift(x_ref=preds_ref_dist, p_val=args.p_val_threshold)
    resultado_preds = cd_preds.predict(preds_drift_dist)
    
    prediction_drift_detected = bool(resultado_preds['data']['is_drift'])
    prediction_p_valor = resultado_preds['data']['p_val'].item()
    prediction_ks_stat = resultado_preds['data']['distance'].item()

    if prediction_drift_detected:
        logger.warning(f"  ¡¡¡ DRIFT DE PREDICCIÓN DETECTADO !!! (p-value: {prediction_p_valor:.6f})")
    else:
        logger.info(f"  No se detectó drift de predicción significativo (p-value: {prediction_p_valor:.6f})")

    # --- 5. Registrar Resultados en W&B Summary ---
    run.summary['prediction_drift_detected'] = prediction_drift_detected
    run.summary['prediction_p_value'] = prediction_p_valor
    run.summary['feature_drift_detected'] = feature_drift_detected
    run.summary['feature_p_value_min'] = feature_p_valor_minimo
    
    logger.info("*" * 50)
    logger.info("Proceso de Detección de Drift finalizado")
    logger.info("*" * 50)


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Simular y detectar Data Drift usando GMM y Alibi-detect")
    parser.add_argument("--model_artifact", type=str, required=True)
    parser.add_argument("--reference_dataset", type=str, required=True)
    parser.add_argument("--gmm_components", type=int, default=3)
    parser.add_argument("--drift_magnitude", type=float, default=0.5)
    parser.add_argument("--n_samples_synthetic", type=int, default=1000)
    parser.add_argument("--p_val_threshold", type=float, default=0.01)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    go(args)