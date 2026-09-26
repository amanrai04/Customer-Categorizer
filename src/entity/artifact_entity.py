"""Dataclasses passed between pipeline stages.

A component never returns raw paths or metrics; it returns one of these
artifacts. The orchestrating pipeline passes the artifact of stage *n* into
stage *n+1*, which makes the hand-off between stages explicit and keeps the
components independent of each other.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataIngestionArtifact:
    """Paths of the train/test split produced by data ingestion."""

    trained_file_path: str
    test_file_path: str


@dataclass(frozen=True)
class DataValidationArtifact:
    """Verdict of the validation stage plus the paths it validated."""

    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass(frozen=True)
class DataTransformationArtifact:
    """Locations of the fitted preprocessor and the transformed matrices."""

    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass(frozen=True)
class ClassificationMetricArtifact:
    """Weighted precision / recall / F1 for a classification model."""

    f1_score: float
    precision_score: float
    recall_score: float


@dataclass(frozen=True)
class ModelTrainerArtifact:
    """Path of the freshly trained model and its metrics on the test split."""

    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """Comparison between the freshly trained model and the promoted one."""

    is_model_accepted: bool
    changed_accuracy: float
    best_model_path: Optional[str]
    trained_model_path: str
    best_model_metric_artifact: Optional[ClassificationMetricArtifact] = None


@dataclass(frozen=True)
class ModelPusherArtifact:
    """Where the accepted model ended up in object storage."""

    bucket_name: str
    s3_model_path: str
