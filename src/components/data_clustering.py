"""Stage 3: turn continuous customer profiles into discrete clusters.

The preprocessed feature matrix is first compressed with PCA, then grouped with
K-Means. The resulting cluster id becomes the label for the supervised model
trained in the next stage, so this step bridges unsupervised and supervised
learning.
"""

import sys
from dataclasses import asdict

from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import ClusteringConfig, PCAConfig
from src.exception import CustomerException
from src.logger import logging


class CreateClusters:
    """Assigns a cluster label to every customer record."""

    def __init__(
        self,
        pca_config: PCAConfig = PCAConfig(),
        clustering_config: ClusteringConfig = ClusteringConfig(),
    ):
        self.pca_config = pca_config
        self.clustering_config = clustering_config

    def get_dataset_using_pca(self, preprocessed_data: DataFrame):
        """Project the feature matrix onto its leading components.

        Args:
            preprocessed_data: Scaled feature matrix.

        Returns:
            The lower dimensional representation used for clustering.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            pca = PCA(**asdict(self.pca_config))
            reduced_dataset = pca.fit_transform(preprocessed_data)

            logging.info(
                f"PCA reduced {preprocessed_data.shape[1]} features down to "
                f"{self.pca_config.n_components}, "
                f"explaining {pca.explained_variance_ratio_.sum():.2%} of the variance"
            )
            return reduced_dataset
        except Exception as error:
            raise CustomerException(error, sys) from error

    def initialize_clustering(self, preprocessed_data: DataFrame) -> DataFrame:
        """Cluster the customers and attach the labels to the input frame.

        Args:
            preprocessed_data: Scaled feature matrix, one row per customer.

        Returns:
            ``preprocessed_data`` with a ``cluster`` column added.

        On Failure:
            Raises :class:`CustomerException`.
        """
        try:
            logging.info("Initializing clustering...")

            reduced_dataset = self.get_dataset_using_pca(preprocessed_data)

            kmeans = KMeans(
                n_clusters=self.clustering_config.n_clusters,
                random_state=self.clustering_config.random_state,
                n_init=10,
            )
            cluster_labels = kmeans.fit_predict(reduced_dataset)

            # Copy so the caller's frame is never modified in place.
            labelled_data = preprocessed_data.copy()
            labelled_data[TARGET_COLUMN] = cluster_labels.astype(int)

            logging.info(
                f"Clustering is done: {self.clustering_config.n_clusters} clusters "
                f"assigned to {len(labelled_data)} records"
            )
            return labelled_data
        except Exception as error:
            raise CustomerException(error, sys) from error
