"""Database identifiers used by the data ingestion component."""

import os

#: MongoDB database that holds the customer records.
DATABASE_NAME: str = os.getenv("MONGODB_DATABASE_NAME", "customer_segmentation")

#: Collection inside ``DATABASE_NAME`` holding the customer records.
COLLECTION_NAME: str = os.getenv("MONGODB_COLLECTION_NAME", "customer_segmentation")
