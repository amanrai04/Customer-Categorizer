"""Turns raw customer input into the dataframe the trained model expects.

Kept free of any storage dependency so it can be imported, and unit tested,
without pulling in the AWS or MongoDB clients.
"""

import sys
from typing import Dict, List, Optional

import pandas as pd

from src.entity.config_entity import PredictionSchemaConfig
from src.exception import CustomerException
from src.logger import logging


class CustomerInputBuilder:
    """Builds a typed, single row feature vector from unstructured input."""

    def __init__(
        self, prediction_schema_config: Optional[PredictionSchemaConfig] = None
    ):
        self.prediction_schema_config = (
            prediction_schema_config or PredictionSchemaConfig()
        )

    def build(self, input_data: List) -> pd.DataFrame:
        """Coerce ``input_data`` into a single row typed dataframe.

        Values arrive as text (an HTML form, a CSV cell, a JSON payload), so
        each one is cast to the dtype declared in the prediction schema. This
        keeps the model's input contract in one place instead of duplicating it
        in every caller.

        Args:
            input_data: Values ordered as
                :meth:`PredictionSchemaConfig.column_names`.

        Returns:
            A one row dataframe with the schema's column names and dtypes.

        On Failure:
            Raises :class:`CustomerException` if a value cannot be cast.
        """
        try:
            column_schema: Dict = self.prediction_schema_config.columns
            input_dataframe = pd.DataFrame(
                [input_data], columns=list(column_schema.keys())
            )

            for column, dtype in column_schema.items():
                input_dataframe[column] = input_dataframe[column].astype(dtype)

            logging.info(
                f"Prepared the input dataframe with the features "
                f"{list(input_dataframe.columns)}"
            )
            return input_dataframe
        except Exception as error:
            raise CustomerException(error, sys) from error
