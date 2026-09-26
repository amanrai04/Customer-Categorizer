"""MongoDB connection helper used by the data ingestion component."""

import os
import sys

import certifi
import pymongo

from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGODB_URL_KEY
from src.exception import CustomerException


class MongoDBClient:
    """Process wide MongoDB client with the requested database pre-selected.

    The connection string is read from the ``MONGO_DB_URL`` environment
    variable, e.g.::

        export MONGO_DB_URL="mongodb+srv://user:password@cluster.mongodb.net"
    """

    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise EnvironmentError(
                        f"Environment variable: {MONGODB_URL_KEY} is not set."
                    )
                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url, tlsCAFile=certifi.where()
                )

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as error:
            raise CustomerException(error, sys) from error
