"""Serving pipeline: turn a raw customer record into a cluster prediction.

The trained model lives in object storage, so every prediction follows the same
three steps: coerce the incoming values into the dtypes declared in
``config/prediction_schema.yaml``, fetch the promoted model, and predict.
"""

import sys
from typing import List

import pandas as pd

from src.data_access.customer_input import CustomerInputBuilder
from src.entity.config_entity import PredictionPipelineConfig
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.s3_estimator import CustomerClusterEstimator


class PredictionPipeline:
    """Scores a single customer against the promoted model."""

    def __init__(self):
        self.input_builder = CustomerInputBuilder()
        self.prediction_config = PredictionPipelineConfig()

    def prepare_input_data(self, input_data: List) -> pd.DataFrame:
        """Return the typed input dataframe for ``input_data``.

        Args:
            input_data: Raw values in prediction schema order.
        """
        logging.info("Entered prepare_input_data method of PredictionPipeline")
        input_dataframe = self.input_builder.build(input_data)
        logging.info("Exited prepare_input_data method of PredictionPipeline")
        return input_dataframe

    def get_trained_model(self) -> CustomerClusterEstimator:
        """Return an estimator bound to the promoted model in the bucket."""
        try:
            return CustomerClusterEstimator(
                bucket_name=self.prediction_config.model_bucket_name,
                model_path=self.prediction_config.model_file_name,
            )
        except Exception as error:
            raise CustomerException(error, sys) from error

    def run_pipeline(self, input_data: List):
        """Run the whole prediction pipeline.

        Args:
            input_data: Raw values in prediction schema order.

        Returns:
            Array of predicted cluster labels.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            logging.info("Entered run_pipeline method of PredictionPipeline")

            input_dataframe = self.prepare_input_data(input_data)
            model = self.get_trained_model()
            predictions = model.predict(input_dataframe)

            logging.info(f"Predicted clusters: {predictions}")
            logging.info("Exited run_pipeline method of PredictionPipeline")
            return predictions
        except Exception as error:
            raise CustomerException(error, sys) from error
