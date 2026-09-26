"""Stage 1 of the training pipeline: pull the raw data and split it.

Reads every customer document from MongoDB, drops the columns declared as noise
in ``config/schema.yaml``, caches the raw export as a "feature store" CSV and
finally writes the train/test split used by every later stage.
"""

import os
import sys
from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.constant.database import COLLECTION_NAME
from src.data_access.customer_data import CustomerData
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataIngestion:
    """Exports the customer collection and prepares the train/test split."""

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

    def export_customers_from_database(self) -> DataFrame:
        """Export the customer collection and cache it as the feature store.

        Returns:
            One row per customer, including the columns listed in
            ``drop_columns`` of ``config/schema.yaml``.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            logging.info(f"Exporting customers from the {COLLECTION_NAME} collection")

            customer_data = CustomerData()
            customer_dataframe = customer_data.export_collection_as_dataframe(
                collection_name=COLLECTION_NAME
            )
            logging.info(f"Exported dataframe with shape {customer_dataframe.shape}")

            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            customer_dataframe.to_csv(feature_store_path, index=False)
            logging.info(f"Cached the raw export at {feature_store_path}")

            return customer_dataframe
        except Exception as error:
            raise CustomerException(error, sys) from error

    def split_data_as_train_test(
        self, dataframe: DataFrame
    ) -> Tuple[str, str]:
        """Persist a train/test split of ``dataframe``.

        Args:
            dataframe: Cleaned customer records.

        Returns:
            The ``(train_path, test_path)`` of the files just written.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )

            ingested_data_dir = self.data_ingestion_config.ingested_data_dir
            os.makedirs(ingested_data_dir, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            logging.info(
                f"Wrote {len(train_set)} training rows to "
                f"{self.data_ingestion_config.training_file_path}"
            )

            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            logging.info(
                f"Wrote {len(test_set)} test rows to "
                f"{self.data_ingestion_config.testing_file_path}"
            )

            logging.info("Exited split_data_as_train_test method of DataIngestion")
            return (
                self.data_ingestion_config.training_file_path,
                self.data_ingestion_config.testing_file_path,
            )
        except Exception as error:
            raise CustomerException(error, sys) from error

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Run the whole ingestion stage.

        Returns:
            A :class:`DataIngestionArtifact` pointing at the train/test files.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion")

        try:
            dataframe = self.export_customers_from_database()

            schema_config = self.utils.read_schema_config_file()
            drop_columns = schema_config.get("drop_columns", [])
            dataframe = dataframe.drop(columns=drop_columns, errors="ignore")
            logging.info(f"Dropped the noise columns {drop_columns}")

            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info("Exited initiate_data_ingestion method of DataIngestion")
            return data_ingestion_artifact
        except Exception as error:
            raise CustomerException(error, sys) from error
