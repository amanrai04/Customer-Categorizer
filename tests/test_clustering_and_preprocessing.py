"""Tests for the clustering step and the preprocessing pipeline."""

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from src.components.data_clustering import CreateClusters
from src.components.data_transformation import (
    MODEL_FEATURES,
    DataTransformation,
    build_features,
)
from src.constant.training_pipeline import OUTLIER_FEATURES, TARGET_COLUMN
from src.entity.config_entity import SimpleImputerConfig
from tests.conftest import make_raw_records


def transformers_of(preprocessor: ColumnTransformer) -> dict:
    """Map each sub-transformer name to its ``(estimator, columns)`` pair."""
    return {
        name: (estimator, columns)
        for name, estimator, columns in preprocessor.transformers
    }


@pytest.fixture
def transformation() -> DataTransformation:
    """A transformation stage wired up with a real imputer config.

    The configs only describe file paths, which these tests never touch, so the
    instance is created without running ``__init__``.
    """
    instance = DataTransformation.__new__(DataTransformation)
    instance.imputer_config = SimpleImputerConfig()
    return instance


class TestBuildPreprocessor:
    def test_splits_numeric_and_skewed_columns(self, transformation):
        preprocessor = transformation.build_preprocessor(MODEL_FEATURES)
        sub_transformers = transformers_of(preprocessor)

        _, numeric_columns = sub_transformers["numeric_pipeline"]
        _, outlier_columns = sub_transformers["outlier_pipeline"]

        assert set(outlier_columns) == set(OUTLIER_FEATURES)
        assert set(numeric_columns) == set(MODEL_FEATURES) - set(OUTLIER_FEATURES)
        assert not set(numeric_columns) & set(outlier_columns)

    def test_output_shape_matches_the_input(self, transformation, raw_records):
        features = build_features(raw_records)
        preprocessor = transformation.build_preprocessor(list(features.columns))

        transformed = preprocessor.fit_transform(features)

        assert transformed.shape == features.shape

    def test_imputes_missing_income_before_scaling(self, transformation, raw_records):
        features = build_features(raw_records)
        features.loc[0, "Income"] = float("nan")

        preprocessor = transformation.build_preprocessor(list(features.columns))
        transformed = preprocessor.fit_transform(features)

        assert np.isfinite(transformed).all()


class TestClustering:
    def test_assigns_the_configured_number_of_clusters(self):
        records = make_raw_records(size=30)
        features = build_features(records)

        clusterer = CreateClusters()
        labelled = clusterer.initialize_clustering(preprocessed_data=features)

        assert TARGET_COLUMN in labelled.columns
        assert labelled[TARGET_COLUMN].nunique() == clusterer.clustering_config.n_clusters
        assert labelled[TARGET_COLUMN].dtype.kind == "i"

    def test_is_deterministic(self):
        def labels_for_run():
            features = build_features(make_raw_records(size=30))
            return CreateClusters().initialize_clustering(
                preprocessed_data=features
            )[TARGET_COLUMN].tolist()

        assert labels_for_run() == labels_for_run()

    def test_does_not_mutate_the_input_frame(self):
        features = build_features(make_raw_records(size=30))
        before = features.copy(deep=True)

        CreateClusters().initialize_clustering(preprocessed_data=features)

        assert list(before.columns) == list(features.columns)

    def test_pca_reduces_to_the_configured_dimensions(self):
        features = build_features(make_raw_records(size=30))
        clusterer = CreateClusters()

        reduced = clusterer.get_dataset_using_pca(features)

        assert reduced.shape == (len(features), clusterer.pca_config.n_components)
        assert np.isfinite(reduced).all()
