import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.train_model.feature_engineering_pipeline import get_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def train_model(xgb_config: dict) -> Pipeline:
    """
    Construye el pipeline del modelo con preprocesador y estimador XGBoost.
    """
    logger.info("Construyendo el pipeline...")
    
    preprocessor = get_preprocessor()

    xgb_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        **xgb_config
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])

    return model_pipeline