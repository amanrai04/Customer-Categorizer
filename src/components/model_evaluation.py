"""Stage 6: decide whether the new model is worth promoting.

The freshly trained model is scored on the test split and compared against the
model currently promoted in object storage. The new model is only pushed when it
beats the incumbent, which keeps a bad training run from overwriting a good
model in production.
"""

import sys
from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import f1_score

from src.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import CustomerException
from src.logger import logging
from src.ml.metric import calculate_metric
from src.ml.model.s3_estimator import CustomerClusterEstimator
from src.utils.main_utils import MainUtils, load_numpy_array_data


@dataclass(frozen=True)
class ModelEvaluationResponse:
    """Side by side comparison of the new model and the promoted model."""

    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    changed_accuracy: float
    best_model_metric_artifact: Optional[ClassificationMetricArtifact]


class ModelEvaluation:
    """Compares the trained model with the model already in the bucket."""

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.utils = MainUtils()

    def get_promoted_model(self) -> Optional[CustomerClusterEstimator]:
        """Return the model currently promoted in the bucket, if there is one."""
        try:
            model_path = self.model_eval_config.s3_model_key_path
            promoted_model = CustomerClusterEstimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=model_path,
            )
            if promoted_model.is_model_present(model_path=model_path):
                return promoted_model
            logging.info("No model has been promoted yet, skipping the comparison")
            return None
        except Exception as error:
            raise CustomerException(error, sys) from error

    def evaluate_model(self) -> ModelEvaluationResponse:
        """Score both models on the test split and compare them.

        The transformed test matrix is already scaled, so the preprocessor
        bundled with each serialized model is deliberately bypassed here:
        scoring the raw estimator keeps the comparison honest.

        Returns:
            A :class:`ModelEvaluationResponse`.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            test_array = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            trained_model = self.utils.load_object(
                self.model_trainer_artifact.trained_model_file_path
            )
            trained_model_f1_score = f1_score(
                y_test, trained_model.trained_model_object.predict(x_test), average="weighted"
            )
            logging.info(f"Trained model weighted f1: {trained_model_f1_score:.4f}")

            best_model = self.get_promoted_model()
            best_model_f1_score = None
            best_model_metric_artifact = None

            if best_model is not None:
                promoted_estimator = best_model.load_model().trained_model_object
                best_model_f1_score = f1_score(
                    y_test, promoted_estimator.predict(x_test), average="weighted"
                )
                best_model_metric_artifact = calculate_metric(
                    promoted_estimator, x_test, y_test
                )
                logging.info(f"Promoted model weighted f1: {best_model_f1_score:.4f}")

            # With no incumbent model anything above zero counts as an improvement.
            baseline_score = best_model_f1_score or 0.0
            changed_accuracy = trained_model_f1_score - baseline_score
            is_model_accepted = (
                changed_accuracy > self.model_eval_config.changed_threshold_score
            )

            response = ModelEvaluationResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                changed_accuracy=changed_accuracy,
                best_model_metric_artifact=best_model_metric_artifact,
            )
            logging.info(f"Model evaluation result: {response}")
            return response
        except Exception as error:
            raise CustomerException(error, sys) from error

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Run the whole evaluation stage.

        Returns:
            A :class:`ModelEvaluationArtifact` whose ``is_model_accepted``
            decides whether the model pusher runs.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            evaluation = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation.is_model_accepted,
                best_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation.changed_accuracy,
                best_model_metric_artifact=evaluation.best_model_metric_artifact,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error
