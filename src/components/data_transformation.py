"""Stage 4: feature engineering, scaling and cluster labelling.

Three things happen here:

1. ``build_features`` turns the raw marketing columns into the 21 model
   features described by ``config/prediction_schema.yaml``. It is a pure
   function so the same transformation can be unit tested and reused from a
   notebook.
2. ``transform_data`` fits a ``ColumnTransformer`` that imputes, then
   standardises the numeric columns and applies a ``PowerTransformer`` to the
   heavily right skewed ones.
3. ``initiate_data_transformation`` glues the two together with the clustering
   step and writes the resulting matrices to disk.
"""

import sys
from dataclasses import asdict
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.components.data_clustering import CreateClusters
from src.constant.training_pipeline import OUTLIER_FEATURES, TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import (
    DataTransformationConfig,
    SimpleImputerConfig,
)
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, read_dataframe, save_numpy_array_data

#: Reference year used to turn ``Year_Birth`` into an age. The campaign data was
#: collected in 2022, so this is the year the recorded ages are relative to.
AGE_REFERENCE_YEAR: int = 2022

#: Ordinal encoding of the education levels, from lowest to highest.
EDUCATION_LEVELS: dict = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4,
}

#: Binary encoding of the marital status. ``1`` means living with a partner.
MARITAL_STATUS_LEVELS: dict = {
    "Married": 1,
    "Together": 1,
    "Absurd": 0,
    "Widow": 0,
    "YOLO": 0,
    "Divorced": 0,
    "Single": 0,
    "Alone": 0,
}

#: Columns that are consumed to build new features and are then discarded.
SOURCE_COLUMNS_DROPPED: list = ["Year_Birth", "Kidhome", "Teenhome"]

#: Raw campaign column -> model feature name.
FEATURE_RENAMES: dict = {
    "Marital_Status": "Marital Status",
    "MntWines": "Wines",
    "MntFruits": "Fruits",
    "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets",
    "MntGoldProds": "Gold",
    "NumWebPurchases": "Web",
    "NumCatalogPurchases": "Catalog",
    "NumStorePurchases": "Store",
    "NumDealsPurchases": "Discount Purchases",
}

#: The model feature vector, in the exact order the trained model expects.
MODEL_FEATURES: list = [
    "Age",
    "Education",
    "Marital Status",
    "Parental Status",
    "Children",
    "Income",
    "Total_Spending",
    "Days_as_Customer",
    "Recency",
    "Wines",
    "Fruits",
    "Meat",
    "Fish",
    "Sweets",
    "Gold",
    "Web",
    "Catalog",
    "Store",
    "Discount Purchases",
    "Total Promo",
    "NumWebVisitsMonth",
]

#: Campaign columns summed up to obtain the total amount spent.
SPENDING_COLUMNS: list = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

#: Campaign columns summed up to obtain the number of accepted promotions.
PROMOTION_COLUMNS: list = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]


def build_features(dataset: DataFrame) -> DataFrame:
    """Build the model feature matrix from a raw customer dataframe.

    The input frame is never modified; a new frame holding exactly
    :data:`MODEL_FEATURES` is returned.

    Args:
        dataset: Raw customer records, i.e. one row per customer with the
            original marketing campaign column names.

    Returns:
        A dataframe with the 21 engineered features.
    """
    features = dataset.copy()

    # Demographics
    features["Age"] = AGE_REFERENCE_YEAR - features["Year_Birth"]
    features["Children"] = features["Kidhome"] + features["Teenhome"]
    features["Parental Status"] = np.where(features["Children"] > 0, 1, 0)

    # Spend and engagement, summed from the original campaign columns.
    features["Total_Spending"] = features[SPENDING_COLUMNS].sum(axis=1)
    features["Total Promo"] = features[PROMOTION_COLUMNS].sum(axis=1)

    # Tenure
    signup_date = pd.to_datetime(features["Dt_Customer"])
    features["Days_as_Customer"] = (datetime.now() - signup_date).dt.days

    # Rename the campaign columns to their model feature names.
    features = features.rename(columns=FEATURE_RENAMES)

    # Encode the two ordinal categorical features.
    features["Education"] = features["Education"].replace(EDUCATION_LEVELS)
    features["Marital Status"] = features["Marital Status"].replace(
        MARITAL_STATUS_LEVELS
    )

    # Keep only the model features, in the canonical order.
    return features[MODEL_FEATURES]


class DataTransformation:
    """Engineers features, fits the preprocessor and labels the clusters."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self.imputer_config = SimpleImputerConfig()
        self.utils = MainUtils()

    def get_new_features(
        self, train_set: DataFrame, test_set: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Apply :func:`build_features` to both splits.

        Returns:
            The engineered ``(train_set, test_set)``.
        """
        train_features = build_features(train_set)
        test_features = build_features(test_set)
        logging.info("Engineered the model features for the train and test splits")
        return train_features, test_features

    def build_preprocessor(self, feature_names: list) -> ColumnTransformer:
        """Create the preprocessing pipeline.

        Regular numeric columns are imputed and standardised, while the skewed
        spend related columns get a :class:`PowerTransformer`, which makes them
        roughly Gaussian and therefore better suited to a linear classifier.

        Args:
            feature_names: Columns of the frame the transformer will be fitted on.
        """
        imputer_kwargs = asdict(self.imputer_config)

        numeric_features = [
            feature for feature in feature_names if feature not in OUTLIER_FEATURES
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(**imputer_kwargs)),
                ("scaler", StandardScaler()),
            ]
        )
        outlier_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(**imputer_kwargs)),
                ("transformer", PowerTransformer(standardize=True)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric_pipeline", numeric_pipeline, numeric_features),
                ("outlier_pipeline", outlier_pipeline, OUTLIER_FEATURES),
            ]
        )

    def transform_data(
        self, train_set: DataFrame, test_set: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """Fit the preprocessor on the train split and apply it to both.

        Args:
            train_set: Engineered training features.
            test_set: Engineered test features.

        Returns:
            The scaled ``(train_set, test_set)``.

        On Failure:
            Raises :class:`CustomerException`.
        """
        logging.info("Entered transform_data method of DataTransformation")

        try:
            preprocessor = self.build_preprocessor(list(train_set.columns))

            preprocessed_train_set = preprocessor.fit_transform(train_set)
            preprocessed_test_set = preprocessor.transform(test_set)

            # The transformed columns are re-labelled with the original feature
            # names on purpose: the ColumnTransformer selects its inputs by name,
            # so the names recorded at fit time must match the ones the
            # prediction schema produces at inference time.
            feature_names = list(train_set.columns)
            preprocessed_train_set = pd.DataFrame(
                preprocessed_train_set, columns=feature_names
            )
            preprocessed_test_set = pd.DataFrame(
                preprocessed_test_set, columns=feature_names
            )

            self.utils.save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )
            logging.info(
                f"Saved the fitted preprocessor to "
                f"{self.data_transformation_config.transformed_object_file_path}"
            )

            logging.info("Exited transform_data method of DataTransformation")
            return preprocessed_train_set, preprocessed_test_set
        except Exception as error:
            raise CustomerException(error, sys) from error

    def attach_cluster_labels(
        self, preprocessed_set: DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster a split and split it into ``(features, labels)``.

        Args:
            preprocessed_set: Scaled features for one split.

        Returns:
            The feature matrix and the cluster labels as two arrays.
        """
        labelled_set = CreateClusters().initialize_clustering(
            preprocessed_data=preprocessed_set
        )
        features = labelled_set.drop(columns=[TARGET_COLUMN])
        labels = labelled_set[TARGET_COLUMN]
        return np.asarray(features), np.asarray(labels)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Run the whole transformation stage.

        Returns:
            A :class:`DataTransformationArtifact` pointing at the fitted
            preprocessor and the two transformed matrices.

        On Failure:
            Raises :class:`CustomerException` if validation failed.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation")

        try:
            if not self.data_validation_artifact.validation_status:
                raise ValueError(
                    "Data validation failed, refusing to transform the dataset."
                )

            train_set = read_dataframe(self.data_ingestion_artifact.trained_file_path)
            test_set = read_dataframe(self.data_ingestion_artifact.test_file_path)

            train_set, test_set = self.get_new_features(train_set, test_set)
            preprocessed_train_set, preprocessed_test_set = self.transform_data(
                train_set, test_set
            )

            x_train, y_train = self.attach_cluster_labels(preprocessed_train_set)
            x_test, y_test = self.attach_cluster_labels(preprocessed_test_set)

            # The target is appended as the last column, which is how the trainer
            # and the evaluator slice the matrices back apart.
            train_array = np.c_[x_train, y_train]
            test_array = np.c_[x_test, y_test]

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, train_array
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, test_array
            )
            logging.info(
                f"Saved the transformed train ({train_array.shape}) and test "
                f"({test_array.shape}) matrices"
            )

            logging.info(
                "Exited initiate_data_transformation method of DataTransformation"
            )

            config = self.data_transformation_config
            return DataTransformationArtifact(
                transformed_object_file_path=config.transformed_object_file_path,
                transformed_train_file_path=config.transformed_train_file_path,
                transformed_test_file_path=config.transformed_test_file_path,
            )
        except Exception as error:
            raise CustomerException(error, sys) from error
