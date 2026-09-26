"""Metric helpers for the classification stages of the pipeline.

All scores are computed with ``average="weighted"`` so clusters of different
sizes contribute proportionally instead of being dominated by the majority
cluster.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.entity.artifact_entity import ClassificationMetricArtifact

#: Cost of a false positive relative to a false negative. Used to report the
#: business cost of a wrong cluster assignment.
FALSE_POSITIVE_COST: int = 10
FALSE_NEGATIVE_COST: int = 500


def calculate_metric(
    model, x: np.ndarray, y: np.ndarray
) -> ClassificationMetricArtifact:
    """Score ``model`` against ground truth ``y``.

    Args:
        model: Any object exposing ``predict``.
        x: Feature matrix.
        y: True labels.

    Returns:
        Weighted precision, recall and F1.
    """
    predictions = model.predict(x)
    return ClassificationMetricArtifact(
        f1_score=f1_score(y, predictions, average="weighted"),
        recall_score=recall_score(y, predictions, average="weighted"),
        precision_score=precision_score(y, predictions, average="weighted"),
    )


def total_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Total cost of the misclassified records.

    Only meaningful for binary problems, where a false positive and a false
    negative can be separated in the confusion matrix.

    Raises:
        ValueError: If the problem is not binary.
    """
    matrix = confusion_matrix(y_true, y_pred)
    if matrix.shape != (2, 2):
        raise ValueError(
            "total_cost is only defined for binary classification, "
            f"got {matrix.shape[0]} classes."
        )

    _, false_positives, false_negatives, _ = matrix.ravel()
    return FALSE_POSITIVE_COST * false_positives + FALSE_NEGATIVE_COST * false_negatives
