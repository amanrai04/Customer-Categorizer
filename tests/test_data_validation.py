"""Tests for the schema validation performed between ingestion and training."""

import pytest

from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import MainUtils
from tests.conftest import make_raw_records


@pytest.fixture
def validator() -> DataValidation:
    """A validator wired to the real schema, without touching any file."""
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path="train.csv", test_file_path="test.csv"
    )
    return DataValidation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_config=DataValidationConfig(),
    )


class TestSchemaValidation:
    def test_accepts_a_matching_dataframe(self, validator):
        records = make_raw_records()
        records = records.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])

        assert validator.validate_schema_columns(records) is True

    def test_rejects_a_missing_column(self, validator):
        records = make_raw_records()
        records = records.drop(columns=["ID", "Z_CostContact", "Z_Revenue", "Complain"])

        assert validator.validate_schema_columns(records) is False

    def test_rejects_an_unexpected_column(self, validator):
        records = make_raw_records()
        records = records.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])
        records["cluster"] = 0

        assert validator.validate_schema_columns(records) is False

    def test_catches_a_rename_with_the_same_column_count(self, validator):
        records = make_raw_records()
        records = records.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])
        records = records.rename(columns={"Complain": "Complaint"})

        assert validator.validate_schema_columns(records) is False

    def test_reports_on_both_splits(self, validator):
        records = make_raw_records().drop(
            columns=["ID", "Z_CostContact", "Z_Revenue"]
        )

        train_status, test_status = validator.validate_dataset_schema_columns(
            train_set=records, test_set=records
        )

        assert train_status is True
        assert test_status is True

    def test_expected_columns_come_from_the_schema_file(self, validator):
        expected = MainUtils().read_schema_config_file()["columns"].keys()

        assert list(validator.expected_columns) == list(expected)
