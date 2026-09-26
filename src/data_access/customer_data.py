"""Read access to the customer records stored in MongoDB."""

import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.configuration.mongo_db_connection import MongoDBClient
from src.constant.database import DATABASE_NAME
from src.exception import CustomerException


class CustomerData:
    """Exports MongoDB collections into pandas dataframes."""

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as error:
            raise CustomerException(error, sys) from error

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Return every document of a collection as a dataframe.

        The Mongo ``_id`` field is dropped and the literal string ``"na"`` is
        converted to ``NaN`` so the frame is ready for pandas.

        Args:
            collection_name: Collection to read from the configured database.
            database_name: Read from a different database when given.

        Returns:
            One row per customer.
        """
        try:
            database = (
                self.mongo_client[database_name]
                if database_name
                else self.mongo_client.database
            )
            dataframe = pd.DataFrame(list(database[collection_name].find()))

            if "_id" in dataframe.columns:
                dataframe = dataframe.drop(columns=["_id"])
            dataframe = dataframe.replace({"na": np.nan})

            return dataframe
        except Exception as error:
            raise CustomerException(error, sys) from error
