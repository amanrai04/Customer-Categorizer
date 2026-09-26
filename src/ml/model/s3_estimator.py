"""Serves the promoted model from object storage and scores incoming data."""

import sys
from typing import Optional

from pandas import DataFrame

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.estimator import CustomerSegmentationModel


class CustomerClusterEstimator:
    """Loads, caches and applies a :class:`CustomerSegmentationModel` from S3.

    The model is downloaded lazily on the first call to :meth:`predict` and then
    kept in memory, so serving many predictions does not re-read S3 every time.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        Args:
            bucket_name: Bucket holding the promoted model.
            model_path: Key of the model object inside the bucket.
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.loaded_model: Optional[CustomerSegmentationModel] = None
        self._storage: Optional[SimpleStorageService] = None

    @property
    def storage(self) -> SimpleStorageService:
        """Return the object storage client, creating it on first use.

        The client is built lazily so that constructing an estimator - and
        therefore importing the app or running the unit tests - does not
        require AWS credentials.
        """
        if self._storage is None:
            self._storage = SimpleStorageService()
        return self._storage

    def is_model_present(self, model_path: Optional[str] = None) -> bool:
        """Return ``True`` when a model already exists in the bucket."""
        try:
            key = model_path or self.model_path
            return self.storage.s3_key_path_available(
                bucket_name=self.bucket_name, s3_key=key
            )
        except Exception as error:
            logging.warning(f"Could not check for an existing model: {error}")
            return False

    def load_model(self) -> CustomerSegmentationModel:
        """Download and unpickle the model from object storage."""
        return self.storage.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """Upload a locally trained model to object storage.

        Args:
            from_file: Path of the serialised model on the local filesystem.
            remove: Delete the local copy after a successful upload.
        """
        try:
            self.storage.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as error:
            raise CustomerException(error, sys) from error

    def predict(self, dataframe: DataFrame):
        """Predict cluster labels, loading the model on first use."""
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as error:
            raise CustomerException(error, sys) from error
