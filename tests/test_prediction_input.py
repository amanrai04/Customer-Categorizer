"""Tests for the serving side: input coercion and application settings."""

import numpy as np
import pytest

from src.constant.application import get_cors_origins
from src.data_access.customer_input import CustomerInputBuilder
from src.entity.config_entity import PredictionSchemaConfig


def schema_values(text_for: dict = None) -> list:
    """Build a valid value for every feature in the prediction schema.

    Args:
        text_for: Optional ``column -> raw string`` overrides, used to test how
            invalid values are handled.
    """
    values = []
    for column, dtype in PredictionSchemaConfig().columns.items():
        if text_for and column in text_for:
            values.append(text_for[column])
        else:
            values.append("1" if dtype == "int" else "1.5")
    return values


@pytest.fixture
def builder() -> CustomerInputBuilder:
    return CustomerInputBuilder()


class TestCustomerInputBuilder:
    def test_produces_a_single_row(self, builder):
        input_dataframe = builder.build(schema_values())

        assert len(input_dataframe) == 1

    def test_uses_the_schema_column_names(self, builder):
        input_dataframe = builder.build(schema_values())

        assert list(input_dataframe.columns) == PredictionSchemaConfig().column_names()

    def test_casts_every_value_to_its_declared_dtype(self, builder):
        column_schema = PredictionSchemaConfig().columns

        input_dataframe = builder.build(schema_values())

        for column, dtype in column_schema.items():
            assert input_dataframe[column].dtype == np.dtype(dtype), column

    def test_casts_whole_string_values_to_numbers(self, builder):
        input_dataframe = builder.build(schema_values())

        assert not any(
            input_dataframe[column].dtype == object
            for column in input_dataframe.columns
        )

    def test_rejects_values_that_do_not_fit_the_dtype(self, builder):
        values = schema_values({"Age": "not-a-number"})

        with pytest.raises(Exception):
            builder.build(values)


class TestPredictionSchemaConfig:
    def test_reads_every_feature(self):
        schema = PredictionSchemaConfig()

        assert len(schema.column_names()) == 21
        assert "Total_Spending" in schema.columns

    def test_only_uses_supported_dtypes(self):
        supported = {"int", "float", "str", "bool"}

        assert set(PredictionSchemaConfig().columns.values()).issubset(supported)


class TestCorsOrigins:
    def test_defaults_to_any_origin(self, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)

        assert get_cors_origins() == ["*"]

    def test_parses_a_comma_separated_list(self, monkeypatch):
        monkeypatch.setenv(
            "CORS_ORIGINS", "https://a.example.com, https://b.example.com"
        )

        assert get_cors_origins() == [
            "https://a.example.com",
            "https://b.example.com",
        ]

    def test_ignores_empty_entries(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com,,")

        assert get_cors_origins() == ["https://a.example.com"]
