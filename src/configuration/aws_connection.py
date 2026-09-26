"""Lazily created, process wide S3 clients.

The boto3 session is created on first use and reused afterwards so every
component in a pipeline run shares a single connection pool.
"""

import os
from typing import Optional

import boto3

from src.constant.env_variable import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_DEFAULT_REGION_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    DEFAULT_REGION,
)


class S3Client:
    """Thin wrapper exposing a shared ``boto3`` S3 resource and client."""

    s3_resource = None
    s3_client = None

    def __init__(self, region_name: Optional[str] = None):
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            region = region_name or os.getenv(
                AWS_DEFAULT_REGION_ENV_KEY, DEFAULT_REGION
            )
            access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            if access_key_id is None:
                raise EnvironmentError(
                    f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set."
                )
            if secret_access_key is None:
                raise EnvironmentError(
                    f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set."
                )

            S3Client.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region,
            )
            S3Client.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region,
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
