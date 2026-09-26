"""Stage 5: train the classifier that predicts a customer's cluster.

The search space lives in ``config/model.yaml`` and is explored by the model
factory, which returns the best candidate together with its cross validated
score. Training only succeeds when that score clears
:attr:`ModelTrainerConfig.expected_accuracy`; the artifact carries the metrics
measured on the real test split rather than a placeholder.
"""

import sys

from neuro_mf import ModelFactory

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.exception import CustomerException
from src.logger import logging
from src.ml.metric import calculate_metric
from src.ml.model.estimator import CustomerSegmentationModel
from src.utils.main_utils import MainUtils, load_numpy_array_data


class ModelTrainer:
    """Selects, fits and persists the cluster classifier."""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def _load_splits(self) -> tuple:
        """Read the transformed matrices and split features from labels.

        Returns:
            ``(x_train, y_train, x_test, y_test)``. The target is the last column
            of each matrix.
        """
        train_array = load_numpy_array_data(
            self.data_transformation_artifact.transformed_train_file_path
        )
        test_array = load_numpy_array_data(
            self.data_transformation_artifact.transformed_test_file_path
        )
        return (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1],
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Run the whole model training stage.

        Returns:
            A :class:`ModelTrainerArtifact` holding the serialised model and the
            metrics measured on the test split.

        On Failure:
            Raises :class:`CustomerException` when no candidate model reaches the
            expected accuracy.
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer")

        try:
            x_train, y_train, x_test, y_test = self._load_splits()

            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_config_file_path
            )
            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            logging.info(
                f"Best model: {type(best_model_detail.best_model).__name__} with a "
                f"score of {best_model_detail.best_score:.4f}"
            )

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise ValueError(
                    "No model reached the expected accuracy of "
                    f"{self.model_trainer_config.expected_accuracy}; best score was "
                    f"{best_model_detail.best_score:.4f}."
                )

            preprocessor = self.utils.load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            segmentation_model = CustomerSegmentationModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model_detail.best_model,
            )

            # The test matrix is already preprocessed, so the raw estimator is
            # scored directly. Wrapping it in the segmentation model would apply
            # the preprocessor a second time.
            metric_artifact = calculate_metric(best_model_detail.best_model, x_test, y_test)
            logging.info(
                f"Test metrics: f1={metric_artifact.f1_score:.4f}, "
                f"precision={metric_artifact.precision_score:.4f}, "
                f"recall={metric_artifact.recall_score:.4f}"
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=segmentation_model,
            )
            logging.info(
                f"Saved the trained model to "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model training artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error
