"""Tests for the feature engineering step."""

import pandas as pd

from src.components.data_transformation import (
    AGE_REFERENCE_YEAR,
    MODEL_FEATURES,
    PROMOTION_COLUMNS,
    SPENDING_COLUMNS,
    build_features,
)
from src.entity.config_entity import PredictionSchemaConfig


class TestBuildFeatures:
    def test_returns_exactly_the_model_features(self, raw_records):
        features = build_features(raw_records)

        assert list(features.columns) == MODEL_FEATURES
        assert len(features) == len(raw_records)

    def test_does_not_mutate_the_input(self, raw_records):
        before = raw_records.copy(deep=True)
        build_features(raw_records)

        pd.testing.assert_frame_equal(raw_records, before)

    def test_age_is_derived_from_the_reference_year(self, raw_records):
        features = build_features(raw_records)

        expected = AGE_REFERENCE_YEAR - raw_records["Year_Birth"]
        assert features["Age"].tolist() == expected.tolist()

    def test_education_is_encoded_ordinally(self, raw_records):
        features = build_features(raw_records)

        assert sorted(features["Education"].unique()) == [0, 1, 2, 3, 4]

    def test_marital_status_is_encoded_as_a_pair(self, raw_records):
        features = build_features(raw_records)

        assert set(features["Marital Status"].unique()).issubset({0, 1})

    def test_children_is_the_household_total(self, raw_records):
        features = build_features(raw_records)

        expected = raw_records["Kidhome"] + raw_records["Teenhome"]
        assert features["Children"].tolist() == expected.tolist()

    def test_parental_status_follows_children(self, raw_records):
        features = build_features(raw_records)

        expected = (features["Children"] > 0).astype(int)
        assert features["Parental Status"].tolist() == expected.tolist()

    def test_total_spending_sums_the_category_columns(self, raw_records):
        features = build_features(raw_records)

        expected = raw_records[SPENDING_COLUMNS].sum(axis=1)
        assert features["Total_Spending"].tolist() == expected.tolist()

    def test_total_promo_sums_the_accepted_campaigns(self, raw_records):
        features = build_features(raw_records)

        expected = raw_records[PROMOTION_COLUMNS].sum(axis=1)
        assert features["Total Promo"].tolist() == expected.tolist()

    def test_days_as_customer_is_positive(self, raw_records):
        features = build_features(raw_records)

        assert (features["Days_as_Customer"] > 0).all()

    def test_tolerates_missing_values(self, raw_records):
        raw_records = raw_records.copy()
        raw_records.loc[0, "Income"] = float("nan")

        features = build_features(raw_records)

        assert features["Income"].isna().sum() == 1


class TestSchemaConsistency:
    """The model is trained on MODEL_FEATURES but served from the YAML schema.

    If the two ever drift apart, predictions silently break, so they are
    asserted to stay in sync.
    """

    def test_model_features_match_the_prediction_schema(self):
        schema_columns = PredictionSchemaConfig().column_names()

        assert MODEL_FEATURES == schema_columns
