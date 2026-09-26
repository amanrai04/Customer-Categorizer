"""Dataclasses describing where each pipeline stage reads from and writes to.

The configuration objects are intentionally declarative: a component receives
its config as a constructor argument (rather than building paths itself), which
makes the data flow between stages explicit and easy to test.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from src.constant import prediction_pipeline
from src.constant.training_pipeline import (
    CLUSTERING_N_CLUSTERS,
    CLUSTERING_RANDOM_STATE,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    DATA_INGESTION_INGESTED_DIR,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    DATA_VALIDATION_INVALID_DIR,
    DATA_VALIDATION_VALID_DIR,
    FILE_NAME,
    MODEL_FILE_NAME,
    MODEL_PUSHER_BUCKET_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    NPY_EXTENSION,
    PCA_N_COMPONENTS,
    PIPELINE_NAME,
    ARTIFACT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME,
    TEST_FILE_NAME,
    TRAIN_FILE_NAME,
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
)
from src.utils.main_utils import MainUtils

#: Single timestamp for the whole process so every stage of one run writes into
#: the same artefact folder.
RUN_TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass(frozen=True)
class TrainingPipelineConfig:
    """Root of the artefact tree for the current run."""

    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(PIPELINE_NAME, ARTIFACT_DIR, RUN_TIMESTAMP)
    timestamp: str = RUN_TIMESTAMP


#: Shared, immutable training configuration for the current process. Every other
#: config below derives its paths from it, so they are all stamped with the same
#: run timestamp.
training_pipeline_config = TrainingPipelineConfig()


@dataclass(frozen=True)
class DataIngestionConfig:
    """Where the raw export and the train/test split are written."""

    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
    )
    ingested_data_dir: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_INGESTED_DIR
    )
    training_file_path: str = os.path.join(
        ingested_data_dir, TRAIN_FILE_NAME
    )
    testing_file_path: str = os.path.join(
        ingested_data_dir, TEST_FILE_NAME
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


@dataclass(frozen=True)
class DataValidationConfig:
    """Where validation verdicts and the drift report are written."""

    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
    )
    valid_data_dir: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_VALID_DIR
    )
    invalid_data_dir: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_INVALID_DIR
    )
    valid_train_file_path: str = os.path.join(valid_data_dir, TRAIN_FILE_NAME)
    valid_test_file_path: str = os.path.join(valid_data_dir, TEST_FILE_NAME)
    invalid_train_file_path: str = os.path.join(invalid_data_dir, TRAIN_FILE_NAME)
    invalid_test_file_path: str = os.path.join(invalid_data_dir, TEST_FILE_NAME)
    drift_report_file_path: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_DRIFT_REPORT_DIR,
        DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    )


@dataclass(frozen=True)
class DataTransformationConfig:
    """Where the fitted preprocessor and the transformed matrices are written."""

    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
    )
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        f"{os.path.splitext(TRAIN_FILE_NAME)[0]}.{NPY_EXTENSION}",
    )
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        f"{os.path.splitext(TEST_FILE_NAME)[0]}.{NPY_EXTENSION}",
    )
    transformed_object_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJECT_FILE_NAME,
    )


@dataclass(frozen=True)
class ClusteringConfig:
    """K-Means settings used to turn customer profiles into discrete clusters."""

    n_clusters: int = CLUSTERING_N_CLUSTERS
    random_state: int = CLUSTERING_RANDOM_STATE


@dataclass(frozen=True)
class PCAConfig:
    """Dimensionality reduction applied before clustering."""

    n_components: int = PCA_N_COMPONENTS
    random_state: int = CLUSTERING_RANDOM_STATE


@dataclass(frozen=True)
class SimpleImputerConfig:
    """Missing values are replaced with a constant so scaling stays stable."""

    strategy: str = "constant"
    fill_value: int = 0


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Where the trained model is written and the accuracy bar it must clear."""

    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: str = os.path.join(
        model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Location of the currently promoted model, used as the baseline."""

    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


@dataclass(frozen=True)
class ModelPusherConfig:
    """Destination the accepted model is uploaded to."""

    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


@dataclass(frozen=True)
class PredictionPipelineConfig:
    """Everything the serving side needs to fetch the promoted model."""

    model_bucket_name: str = prediction_pipeline.MODEL_BUCKET_NAME
    model_file_name: str = prediction_pipeline.PREDICTION_MODEL_FILE_NAME


@dataclass(frozen=True)
class PredictionSchemaConfig:
    """Feature name to dtype mapping expected by the trained model."""

    prediction_schema: Dict = field(
        default_factory=lambda: MainUtils().read_yaml_file(
            prediction_pipeline.PRED_SCHEMA_FILE_PATH
        )
    )

    @property
    def columns(self) -> Dict:
        return self.prediction_schema["columns"]

    def column_names(self) -> list:
        """Ordered feature names, i.e. the order the model was trained on."""
        return list(self.prediction_schema["columns"].keys())
