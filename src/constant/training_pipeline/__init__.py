"""Constants that describe the layout of the training pipeline.

Every artefact produced by a training run lives under::

    <project_root>/<PIPELINE_NAME>/<ARTIFACT_DIR>/<TIMESTAMP>/<component>/...

Keeping the paths here (instead of inline in the components) means the
directory layout can be changed in exactly one place.
"""

import os

from src.constant.s3_bucket import TRAINING_BUCKET_NAME

# --------------------------------------------------------------------------- #
# Pipeline identity
# --------------------------------------------------------------------------- #

#: Human readable project name. Also the root folder for artefacts and logs.
PIPELINE_NAME: str = "customer_segmentation"

#: Label attached to the cluster produced by the clustering component. Used as
#: the supervised learning target from the clustering step onwards.
TARGET_COLUMN: str = "cluster"

# --------------------------------------------------------------------------- #
# Artefact locations
# --------------------------------------------------------------------------- #

ARTIFACT_DIR: str = "artifact"
LOG_DIR: str = "logs"
LOG_FILE: str = "customer_segmentation.log"

# --------------------------------------------------------------------------- #
# Common file names
# --------------------------------------------------------------------------- #

FILE_NAME: str = "customer.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
NPY_EXTENSION: str = "npy"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
MODEL_FILE_NAME: str = "model.pkl"

# --------------------------------------------------------------------------- #
# Configuration files
# --------------------------------------------------------------------------- #

CONFIG_DIR: str = "config"
SCHEMA_FILE_PATH: str = os.path.join(CONFIG_DIR, "schema.yaml")
PREDICTION_SCHEMA_FILE_PATH: str = os.path.join(CONFIG_DIR, "prediction_schema.yaml")
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join(CONFIG_DIR, "model.yaml")

# --------------------------------------------------------------------------- #
# Data ingestion
# --------------------------------------------------------------------------- #

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# --------------------------------------------------------------------------- #
# Data validation
# --------------------------------------------------------------------------- #

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# --------------------------------------------------------------------------- #
# Data transformation
# --------------------------------------------------------------------------- #

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

#: Columns that are heavily right skewed. They get a ``PowerTransformer``
#: instead of the plain standardisation used for the remaining numeric columns.
OUTLIER_FEATURES: list = [
    "Wines",
    "Fruits",
    "Meat",
    "Fish",
    "Sweets",
    "Gold",
    "Age",
    "Total_Spending",
]

# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

DATA_CLUSTERING_DIR_NAME: str = "data_clustering"
CLUSTERING_N_CLUSTERS: int = 3
CLUSTERING_RANDOM_STATE: int = 42
PCA_N_COMPONENTS: int = 2

# --------------------------------------------------------------------------- #
# Model trainer
# --------------------------------------------------------------------------- #

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

# --------------------------------------------------------------------------- #
# Model pusher
# --------------------------------------------------------------------------- #

MODEL_PUSHER_BUCKET_NAME: str = TRAINING_BUCKET_NAME
