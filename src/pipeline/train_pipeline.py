"""Orchestrates the seven stages of the training pipeline.

::

    ingestion -> validation -> transformation -> training -> evaluation -> pushing

Each stage is wrapped in its own method so it can be run, and re-run, in
isolation. Stages hand their outputs to the next one through the artifact
dataclasses in :mod:`src.entity.artifact_entity`.
"""

import sys
from typing import Optional

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
)
from src.exception import CustomerException
from src.logger import logging


class TrainPipeline:
    """Runs the full retraining cycle end to end."""

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Export the customers and split them into train and test."""
        logging.info("Entered start_data_ingestion method of TrainPipeline")

        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Exited start_data_ingestion method of TrainPipeline")
            return data_ingestion_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Validate the ingested split against the schema."""
        logging.info("Entered start_data_validation method of TrainPipeline")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Exited start_data_validation method of TrainPipeline")
            return data_validation_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """Engineer features, fit the preprocessor and label the clusters."""
        logging.info("Entered start_data_transformation method of TrainPipeline")

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info("Exited start_data_transformation method of TrainPipeline")
            return data_transformation_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Train the cluster classifier on the transformed data."""
        logging.info("Entered start_model_trainer method of TrainPipeline")

        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("Exited start_model_trainer method of TrainPipeline")
            return model_trainer_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        """Compare the trained model with the promoted one."""
        logging.info("Entered start_model_evaluation method of TrainPipeline")

        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logging.info("Exited start_model_evaluation method of TrainPipeline")
            return model_evaluation_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error

    def start_model_pusher(
        self, model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelPusherArtifact:
        """Publish the accepted model to object storage."""
        logging.info("Entered start_model_pusher method of TrainPipeline")

        try:
            model_pusher = ModelPusher(
                model_trainer_artifact=model_trainer_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Exited start_model_pusher method of TrainPipeline")
            return model_pusher_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error

    def run_pipeline(self) -> Optional[ModelPusherArtifact]:
        """Run every stage in order.

        The model is only pushed when evaluation accepts it, so a regression
        never reaches the bucket.

        Returns:
            The :class:`ModelPusherArtifact` for the promoted model, or ``None``
            when the newly trained model was rejected.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered run_pipeline method of TrainPipeline")

        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact,
            )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info(
                    "The trained model did not beat the promoted model by more than "
                    f"{self.model_evaluation_config.changed_threshold_score}, "
                    "skipping the model push"
                )
                return None

            return self.start_model_pusher(
                model_trainer_artifact=model_trainer_artifact
            )
        except Exception as error:
            raise CustomerException(error, sys) from error
        finally:
            logging.info(f"Exited run_pipeline method of {type(self).__name__}")
