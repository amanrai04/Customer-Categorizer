"""Stage 7: publish the accepted model to object storage.

Only reached when :class:`~src.components.model_evaluation.ModelEvaluation`
accepted the new model, so the bucket always holds the best model seen so far.
"""

import sys

from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.s3_estimator import CustomerClusterEstimator


class ModelPusher:
    """Uploads the trained model to the model bucket."""

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.model_registry = CustomerClusterEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Run the whole model publishing stage.

        Returns:
            A :class:`ModelPusherArtifact` describing where the model landed.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher")

        try:
            self.model_registry.save_model(
                from_file=self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(
                f"Uploaded the model to s3://{self.model_pusher_config.bucket_name}/"
                f"{self.model_pusher_config.s3_model_key_path}"
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher")
            return model_pusher_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error
