"""Object storage operations used by the model pusher and the predictor.

Only the handful of operations the pipeline needs are exposed, each of them
wrapping the equivalent boto3 call and converting failures into
:class:`~src.exception.CustomerException`.
"""

import os
import pickle
import sys
from io import StringIO
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd
from botocore.exceptions import ClientError

from src.configuration.aws_connection import S3Client
from src.exception import CustomerException
from src.logger import logging

if TYPE_CHECKING:  # pragma: no cover - import only needed by type checkers
    from mypy_boto3_s3.service_resource import Bucket


class SimpleStorageService:
    """Small, task oriented wrapper around the S3 resource API."""

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def get_bucket(self, bucket_name: str) -> "Bucket":
        """Return the bucket object for ``bucket_name``."""
        logging.info(f"Entered get_bucket of SimpleStorageService for {bucket_name}")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info(f"Exited get_bucket of SimpleStorageService for {bucket_name}")
            return bucket
        except Exception as error:
            raise CustomerException(error, sys) from error

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """Return ``True`` when at least one object matches ``s3_key``."""
        try:
            bucket = self.get_bucket(bucket_name)
            matches = list(bucket.objects.filter(Prefix=s3_key))
            return len(matches) > 0
        except Exception as error:
            raise CustomerException(error, sys) from error

    def get_file_object(
        self, filename: str, bucket_name: str
    ) -> Union[List[object], object]:
        """Return the object(s) stored under ``filename``.

        A single match is returned unwrapped, which keeps the common case simple
        for callers such as :meth:`load_model`.
        """
        logging.info(f"Entered get_file_object of SimpleStorageService for {filename}")

        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = list(bucket.objects.filter(Prefix=filename))
            file_object = file_objects[0] if len(file_objects) == 1 else file_objects

            logging.info(f"Exited get_file_object of SimpleStorageService for {filename}")
            return file_object
        except Exception as error:
            raise CustomerException(error, sys) from error

    @staticmethod
    def read_object(
        object_name: object,
        decode: bool = True,
        make_readable: bool = False,
    ) -> Union[StringIO, str, bytes]:
        """Read the body of an S3 object.

        Args:
            object_name: Object returned by :meth:`get_file_object`.
            decode: Decode the body as UTF-8 text.
            make_readable: Wrap the body in a ``StringIO`` buffer.

        Returns:
            The object body as text, bytes or a ``StringIO`` buffer.
        """
        logging.info("Entered read_object of SimpleStorageService")

        try:
            body = object_name.get()["Body"].read()
            content = body.decode() if decode else body
            result = StringIO(content) if make_readable else content

            logging.info("Exited read_object of SimpleStorageService")
            return result
        except Exception as error:
            raise CustomerException(error, sys) from error

    def load_model(
        self, model_name: str, bucket_name: str, model_dir: Optional[str] = None
    ) -> object:
        """Download and unpickle a model from object storage."""
        logging.info(f"Entered load_model of SimpleStorageService for {model_name}")

        try:
            model_key = model_name if model_dir is None else f"{model_dir}/{model_name}"
            model_file = self.get_file_object(model_key, bucket_name)
            model = pickle.loads(self.read_object(model_file, decode=False))

            logging.info(f"Exited load_model of SimpleStorageService for {model_name}")
            return model
        except Exception as error:
            raise CustomerException(error, sys) from error

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """Create a zero byte marker object so ``folder_name/`` shows up as a folder."""
        logging.info(f"Entered create_folder of SimpleStorageService for {folder_name}")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as error:
            if error.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/")
            else:
                raise CustomerException(error, sys) from error

        logging.info(f"Exited create_folder of SimpleStorageService for {folder_name}")

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ) -> None:
        """Upload a local file, optionally deleting the local copy afterwards."""
        logging.info(
            f"Entered upload_file of SimpleStorageService: {from_filename} -> {to_filename}"
        )

        try:
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(
                f"Uploaded {from_filename} to {to_filename} in bucket {bucket_name}"
            )

            if remove:
                os.remove(from_filename)
                logging.info("Local copy removed after upload")
            else:
                logging.info("Local copy kept after upload")

            logging.info("Exited upload_file of SimpleStorageService")
        except Exception as error:
            raise CustomerException(error, sys) from error

    def upload_df_as_csv(
        self,
        data_frame: pd.DataFrame,
        local_filename: str,
        bucket_filename: str,
        bucket_name: str,
    ) -> None:
        """Serialise a dataframe to CSV and upload it."""
        logging.info("Entered upload_df_as_csv of SimpleStorageService")

        try:
            data_frame.to_csv(local_filename, index=False, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logging.info("Exited upload_df_as_csv of SimpleStorageService")
        except Exception as error:
            raise CustomerException(error, sys) from error

    def get_df_from_object(self, object_: object) -> pd.DataFrame:
        """Read an S3 object holding a CSV file into a dataframe."""
        logging.info("Entered get_df_from_object of SimpleStorageService")

        try:
            content = self.read_object(object_, make_readable=True)
            dataframe = pd.read_csv(content, na_values="na")
            logging.info("Exited get_df_from_object of SimpleStorageService")
            return dataframe
        except Exception as error:
            raise CustomerException(error, sys) from error

    def read_csv(self, filename: str, bucket_name: str) -> pd.DataFrame:
        """Read a CSV file straight out of object storage into a dataframe."""
        logging.info(f"Entered read_csv of SimpleStorageService for {filename}")

        try:
            csv_object = self.get_file_object(filename, bucket_name)
            dataframe = self.get_df_from_object(csv_object)
            logging.info(f"Exited read_csv of SimpleStorageService for {filename}")
            return dataframe
        except Exception as error:
            raise CustomerException(error, sys) from error
