"""Stage 2 of the training pipeline: sanity check the ingested data.

Two checks are run on the train/test split:

* **Schema check** - the columns match ``config/schema.yaml`` exactly, so a
  silently renamed or missing column cannot reach the model. A failure here
  stops the pipeline.
* **Drift check** - the test split is compared against the train split with
  :mod:`src.components.data_drift` and a report is written for inspection.
  Drift is advisory: a random split of a homogeneous population is expected to
  trip the statistical test now and then, so it is logged as a warning and
  surfaced in the report rather than blocking the run.
"""

import sys
from typing import List, Tuple

from pandas import DataFrame

from src.components.data_drift import detect_drift
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, read_dataframe, write_yaml_file


class DataValidation:
    """Validates the ingested split and reports drift between train and test."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self.utils = MainUtils()
        self.schema_config = self.utils.read_schema_config_file()

    @property
    def expected_columns(self) -> List[str]:
        """Columns the pipeline is designed to consume."""
        return list(self.schema_config["columns"].keys())

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        """Check that ``dataframe`` has exactly the expected columns.

        Args:
            dataframe: Split to validate.

        Returns:
            ``True`` when the columns match ``config/schema.yaml``.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            expected = set(self.expected_columns)
            actual = set(dataframe.columns)

            missing = expected - actual
            unexpected = actual - expected
            is_valid = not missing and not unexpected

            if is_valid:
                logging.info(
                    f"Schema check passed for all {len(expected)} expected columns"
                )
            else:
                if missing:
                    logging.warning(f"Columns missing from the dataset: {sorted(missing)}")
                if unexpected:
                    logging.warning(
                        f"Unexpected columns in the dataset: {sorted(unexpected)}"
                    )

            return is_valid
        except Exception as error:
            raise CustomerException(error, sys) from error

    def validate_dataset_schema_columns(
        self, train_set: DataFrame, test_set: DataFrame
    ) -> Tuple[bool, bool]:
        """Run the schema check on both splits.

        Returns:
            ``(train_is_valid, test_is_valid)``.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered validate_dataset_schema_columns method of DataValidation")

        try:
            train_schema_status = self.validate_schema_columns(train_set)
            logging.info(f"Validated the schema of the train set: {train_schema_status}")

            test_schema_status = self.validate_schema_columns(test_set)
            logging.info(f"Validated the schema of the test set: {test_schema_status}")

            return train_schema_status, test_schema_status
        except Exception as error:
            raise CustomerException(error, sys) from error

    def detect_dataset_drift(
        self, reference_df: DataFrame, current_df: DataFrame
    ) -> bool:
        """Compare two splits and persist a drift report.

        Args:
            reference_df: Baseline split, normally the training data.
            current_df: Split to compare against the baseline.

        Returns:
            ``True`` when too many columns have drifted.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            report = detect_drift(reference_df, current_df)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report.to_dict(),
            )

            logging.info(
                f"Drift report: {report.n_drifted_features}/{report.n_features} "
                f"columns drifted -> {report.drifted_columns}"
            )
            logging.info(
                f"Drift report written to {self.data_validation_config.drift_report_file_path}"
            )

            return report.dataset_drift
        except Exception as error:
            raise CustomerException(error, sys) from error

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Run the whole validation stage.

        Returns:
            A :class:`DataValidationArtifact` whose ``validation_status`` is
            ``True`` only when both splits match the schema. Detected drift is
            logged as a warning and recorded in the drift report, but does not
            fail the run.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered initiate_data_validation method of DataValidation")

        try:
            train_df = read_dataframe(self.data_ingestion_artifact.trained_file_path)
            test_df = read_dataframe(self.data_ingestion_artifact.test_file_path)

            drift_detected = self.detect_dataset_drift(train_df, test_df)
            train_is_valid, test_is_valid = self.validate_dataset_schema_columns(
                train_set=train_df, test_set=test_df
            )

            if drift_detected:
                logging.warning(
                    "Drift was detected between the train and test splits. The run "
                    f"continues, but review {self.data_validation_config.drift_report_file_path}"
                )

            validation_status = train_is_valid and test_is_valid
            logging.info(
                f"Validation finished: train={train_is_valid}, test={test_is_valid}, "
                f"drift={drift_detected} -> {validation_status}"
            )

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as error:
            raise CustomerException(error, sys) from error
