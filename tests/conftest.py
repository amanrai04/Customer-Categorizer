"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow ``pytest`` to be run from anywhere inside the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Raw marketing campaign columns, i.e. what a customer record looks like
#: straight out of the database.
RAW_COLUMNS = [
    "ID",
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "Z_CostContact",
    "Z_Revenue",
    "Response",
]


def make_raw_records(size: int = 6) -> pd.DataFrame:
    """Build a small, valid raw customer dataframe for tests.

    Args:
        size: Number of customer records to generate.

    Returns:
        A dataframe with :data:`RAW_COLUMNS`.
    """
    educations = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    marital_statuses = ["Single", "Married", "Together", "Divorced", "Widow"]

    records = pd.DataFrame(
        {
            "ID": range(size),
            "Year_Birth": [1960 + index for index in range(size)],
            "Education": [educations[index % len(educations)] for index in range(size)],
            "Marital_Status": [
                marital_statuses[index % len(marital_statuses)] for index in range(size)
            ],
            "Income": [30000.0 + 1000 * index for index in range(size)],
            "Kidhome": [index % 3 for index in range(size)],
            "Teenhome": [index % 2 for index in range(size)],
            "Dt_Customer": [f"2018-0{index % 9 + 1}-15" for index in range(size)],
            "Recency": [10 + index for index in range(size)],
            "MntWines": [100 + index for index in range(size)],
            "MntFruits": [10 + index for index in range(size)],
            "MntMeatProducts": [200 + index for index in range(size)],
            "MntFishProducts": [20 + index for index in range(size)],
            "MntSweetProducts": [5 + index for index in range(size)],
            "MntGoldProds": [50 + index for index in range(size)],
            "NumDealsPurchases": [index % 4 for index in range(size)],
            "NumWebPurchases": [index % 5 for index in range(size)],
            "NumCatalogPurchases": [index % 6 for index in range(size)],
            "NumStorePurchases": [index % 7 for index in range(size)],
            "NumWebVisitsMonth": [2 + index for index in range(size)],
            "AcceptedCmp1": [index % 2 for index in range(size)],
            "AcceptedCmp2": [index % 2 for index in range(size)],
            "AcceptedCmp3": [0 for _ in range(size)],
            "AcceptedCmp4": [0 for _ in range(size)],
            "AcceptedCmp5": [0 for _ in range(size)],
            "Complain": [0 for _ in range(size)],
            "Z_CostContact": [value / 100 for value in range(size)],
            "Z_Revenue": [value / 10 for value in range(size)],
            "Response": [index % 2 for index in range(size)],
        }
    )
    return records


@pytest.fixture
def raw_records() -> pd.DataFrame:
    """A small dataframe shaped like the raw customer records."""
    return make_raw_records()
