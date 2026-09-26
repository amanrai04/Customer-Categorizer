"""End-to-end test of the training pipeline with MongoDB and S3 stubbed out.

This exercises the real component wiring - ingestion, validation, feature
engineering, clustering, grid search, evaluation and promotion - without
needing a live database, AWS credentials or network access.
"""

import os
import pickle
import shutil
from unittest.mock import patch

import pytest
from pandas import DataFrame

from src.constant.s3_bucket import TRAINING_BUCKET_NAME
from src.constant.training_pipeline import (
    ARTIFACT_DIR,
    MODEL_FILE_NAME,
    PIPELINE_NAME,
)
from src.data_access.customer_data import CustomerData
from src.entity.config_entity import PredictionSchemaConfig
from src.ml.model.s3_estimator import CustomerClusterEstimator
from src.pipeline.train_pipeline import TrainPipeline
from tests.conftest import make_raw_records

ARTIFACT_ROOT = os.path.join(PIPELINE_NAME, ARTIFACT_DIR)


@pytest.fixture(autouse=True)
def mongo_url(monkeypatch):
    """A configured connection string; nothing actually connects to it."""
    monkeypatch.setenv("MONGO_DB_URL", "mongodb://localhost:27017")


@pytest.fixture
def object_store():
    """Stand in for S3 with an in-memory mapping of bucket/key -> file path."""
    store = {}

    def fake_save(self, from_file, remove=False):
        store[(self.bucket_name, self.model_path)] = from_file

    def fake_is_present(self, model_path=None):
        return (self.bucket_name, model_path or self.model_path) in store

    def fake_load(self):
        with open(store[(self.bucket_name, self.model_path)], "rb") as handle:
            return pickle.load(handle)

    with patch.object(CustomerClusterEstimator, "save_model", fake_save), \
         patch.object(CustomerClusterEstimator, "is_model_present", fake_is_present), \
         patch.object(CustomerClusterEstimator, "load_model", fake_load):
        yield store


@pytest.fixture
def mongo_collection():
    """Stand in for MongoDB by serving generated customer records."""
    records = DataFrame(make_raw_records(size=200))

    with patch.object(
        CustomerData,
        "export_collection_as_dataframe",
        lambda self, collection_name, database_name=None: records.copy(),
    ):
        yield records


@pytest.fixture(autouse=True)
def clean_artifacts():
    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)


def list_artifacts() -> list:
    """Return the relative path of every file produced by a training run."""
    written = []
    for root, _directories, files in os.walk(ARTIFACT_ROOT):
        written.extend(os.path.relpath(os.path.join(root, name)) for name in files)
    return written


class TestTrainPipeline:
    def test_accepted_model_is_pushed_to_object_storage(
        self, mongo_collection, object_store
    ):
        result = TrainPipeline().run_pipeline()

        assert result is not None
        assert result.bucket_name == TRAINING_BUCKET_NAME
        assert (TRAINING_BUCKET_NAME, MODEL_FILE_NAME) in object_store

    def test_artifacts_are_written_outside_the_source_package(
        self, mongo_collection, object_store
    ):
        TrainPipeline().run_pipeline()

        assert os.path.isdir(ARTIFACT_ROOT)
        assert not os.path.exists(os.path.join("src", ARTIFACT_DIR))

    def test_run_writes_every_expected_file(self, mongo_collection, object_store):
        TrainPipeline().run_pipeline()

        written = list_artifacts()

        assert any(path.endswith("customer.csv") for path in written)
        assert any(path.endswith("train.csv") for path in written)
        assert any(path.endswith("test.csv") for path in written)
        assert any(path.endswith("train.npy") for path in written)
        assert any(path.endswith("test.npy") for path in written)
        assert any(path.endswith("preprocessing.pkl") for path in written)
        assert any(path.endswith("report.yaml") for path in written)
        assert any(path.endswith(MODEL_FILE_NAME) for path in written)

    def test_promoted_model_scores_a_prediction(self, mongo_collection, object_store):
        TrainPipeline().run_pipeline()

        estimator = CustomerClusterEstimator(
            bucket_name=TRAINING_BUCKET_NAME, model_path=MODEL_FILE_NAME
        )
        model = estimator.load_model()

        # Any plausible record will do: the point is that the pickled
        # preprocessor and classifier work together again after a round trip.
        record = {
            column: 1 for column in PredictionSchemaConfig().columns
        }
        prediction = model.predict(DataFrame([record]))

        assert len(prediction) == 1
        assert int(prediction[0]) in {0, 1, 2}
