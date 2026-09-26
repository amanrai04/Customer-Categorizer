"""The estimator that is actually serialised and deployed.

It bundles the fitted preprocessor together with the trained classifier so a
single pickle contains everything needed to turn a raw customer record into a
cluster prediction. :class:`~src.ml.model.s3_estimator.CustomerClusterEstimator`
downloads and calls it.
"""

import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import CustomerException
from src.logger import logging


class CustomerSegmentationModel:
    """Preprocessor + classifier pair used for inference."""

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """Transform ``dataframe`` with the fitted preprocessor and predict.

        Args:
            dataframe: Raw customer features, one row per customer.

        Returns:
            Array of predicted cluster labels.
        """
        logging.info("Entered predict method of CustomerSegmentationModel")

        try:
            transformed_features = self.preprocessing_object.transform(dataframe)
            predictions = self.trained_model_object.predict(transformed_features)

            logging.info("Exited predict method of CustomerSegmentationModel")
            return predictions
        except Exception as error:
            raise CustomerException(error, sys) from error

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
