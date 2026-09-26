"""Constants for the serving prediction pipeline."""

from src.constant.s3_bucket import TRAINING_BUCKET_NAME
from src.constant.training_pipeline import (
    MODEL_FILE_NAME,
    PREDICTION_SCHEMA_FILE_PATH,
)

#: Schema describing the feature vector expected by the trained model.
PRED_SCHEMA_FILE_PATH: str = PREDICTION_SCHEMA_FILE_PATH

#: Bucket the promoted model is served from, and the key it is stored under.
MODEL_BUCKET_NAME: str = TRAINING_BUCKET_NAME
PREDICTION_MODEL_FILE_NAME: str = MODEL_FILE_NAME
