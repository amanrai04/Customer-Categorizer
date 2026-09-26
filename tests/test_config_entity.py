"""Tests for the configuration objects that define the artefact layout."""

from src.entity.artifact_entity import ClassificationMetricArtifact
from src.entity.config_entity import (
    ClusteringConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    SimpleImputerConfig,
    TrainingPipelineConfig,
    training_pipeline_config,
)


class TestArtifactLayout:
    def test_every_artefact_lives_under_the_project_folder(self):
        config = TrainingPipelineConfig()

        assert config.pipeline_name == "customer_segmentation"
        assert config.artifact_dir.startswith(config.pipeline_name)
        # The bug this guards against: artefacts used to be written inside the
        # source package itself.
        assert not config.artifact_dir.startswith("src")

    def test_all_stages_share_the_same_run_timestamp(self):
        timestamp = training_pipeline_config.timestamp

        for config in (
            DataIngestionConfig(),
            DataValidationConfig(),
            DataTransformationConfig(),
            ModelTrainerConfig(),
        ):
            assert timestamp in config.__dict__.values() or any(
                timestamp in str(value) for value in config.__dict__.values()
            )

    def test_ingestion_paths_are_nested_correctly(self):
        config = DataIngestionConfig()

        assert config.training_file_path.startswith(config.ingested_data_dir)
        assert config.testing_file_path.startswith(config.ingested_data_dir)
        assert config.training_file_path.endswith("train.csv")
        assert config.testing_file_path.endswith("test.csv")

    def test_transformed_files_use_the_numpy_extension(self):
        config = DataTransformationConfig()

        assert config.transformed_train_file_path.endswith("train.npy")
        assert config.transformed_test_file_path.endswith("test.npy")
        assert config.transformed_object_file_path.endswith("preprocessing.pkl")

    def test_drift_report_is_a_yaml_file(self):
        config = DataValidationConfig()

        assert config.drift_report_file_path.endswith("report.yaml")

    def test_model_bucket_matches_the_training_bucket(self):
        from src.constant.s3_bucket import TRAINING_BUCKET_NAME

        assert ModelPusherConfig().bucket_name == TRAINING_BUCKET_NAME


class TestConfigsAreImmutable:
    def test_paths_cannot_be_rewritten_by_a_component(self):
        config = DataIngestionConfig()
        original = config.training_file_path

        try:
            config.training_file_path = "somewhere/else.csv"
        except Exception:
            pass

        assert config.training_file_path == original


class TestModelParameters:
    def test_imputer_uses_a_constant_fill_value(self):
        config = SimpleImputerConfig()

        assert config.strategy == "constant"
        assert config.fill_value == 0

    def test_clustering_uses_three_segments(self):
        config = ClusteringConfig()

        assert config.n_clusters == 3
        assert config.random_state == 42


class TestArtifactEntities:
    def test_metrics_are_value_objects(self):
        first = ClassificationMetricArtifact(f1_score=0.9, precision_score=0.9, recall_score=0.9)
        second = ClassificationMetricArtifact(f1_score=0.9, precision_score=0.9, recall_score=0.9)

        assert first == second
