"""Drift detection between two splits of the customer dataset.

The training split is the reference distribution and the test split is the
current one. For every shared numeric column two independent signals are
computed:

* **Population Stability Index (PSI)** - compares the two distributions across
  fixed quantile buckets. A PSI above :data:`PSI_DRIFT_THRESHOLD` means the
  share of customers falling into a bucket changed noticeably.
* **Kolmogorov-Smirnov statistic** - the largest gap between the two empirical
  CDFs, with the p-value telling us whether that gap is distinguishable from
  sampling noise.

A column is flagged as drifted when either signal fires, and the split as a
whole is drifted when the share of drifted columns exceeds
:data:`MAX_DRIFTED_COLUMN_RATIO`. Both signals are reported so a reviewer can
see *why* a column was flagged before deciding whether it matters.

Only ``numpy`` and ``scipy`` are needed, so drift detection works on any
supported Python without a heavyweight dependency.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import ks_2samp

#: Number of quantile buckets used by the PSI calculation.
PSI_BUCKETS: int = 10

#: PSI above which a column is considered drifted. 0.1 is small, 0.25 is
#: significant; the usual convention is to act somewhere in between.
PSI_DRIFT_THRESHOLD: float = 0.2

#: Share of drifted columns above which the dataset is considered drifted.
MAX_DRIFTED_COLUMN_RATIO: float = 0.5

#: Significance level for the KS test. Above this p-value a difference is
#: treated as noise rather than drift.
KS_SIGNIFICANCE_LEVEL: float = 0.05


@dataclass(frozen=True)
class ColumnDrift:
    """Drift verdict and the evidence behind it for a single column."""

    column: str
    psi: float
    ks_statistic: float
    p_value: float
    is_drifted: bool

    def to_dict(self) -> Dict:
        return {
            "column": self.column,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "p_value": float(f"{self.p_value:.6g}"),
            "is_drifted": self.is_drifted,
        }


@dataclass(frozen=True)
class DriftReport:
    """Drift verdict for a whole dataset."""

    n_features: int
    n_drifted_features: int
    drifted_columns: List[str]
    dataset_drift: bool
    per_column: List[ColumnDrift]

    def to_dict(self) -> Dict:
        return {
            "n_features": self.n_features,
            "n_drifted_features": self.n_drifted_features,
            "drifted_columns": self.drifted_columns,
            "dataset_drift": self.dataset_drift,
            "columns": [column.to_dict() for column in self.per_column],
        }


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_buckets: int = PSI_BUCKETS
) -> float:
    """Compare two samples across quantile buckets of the reference.

    Args:
        reference: Values of the baseline split.
        current: Values of the split being checked.
        n_buckets: Number of quantile buckets.

    Returns:
        The PSI. ``0`` means identical distributions.

    Raises:
        ValueError: If ``n_buckets`` is smaller than two.
    """
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be at least 2, got {n_buckets}")

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size == 0:
        return 0.0

    # np.histogram needs monotonically increasing edges and covers the whole
    # range, so the outer edges are taken from the data itself.
    edges = np.concatenate([[-np.inf], edges, [np.inf]])

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    # A zero share would make the ratio infinite; smoothing keeps PSI finite
    # while still being large enough to flag the shift.
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def detect_column_drift(
    column: str,
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = PSI_DRIFT_THRESHOLD,
    significance_level: float = KS_SIGNIFICANCE_LEVEL,
) -> ColumnDrift:
    """Measure drift for one column.

    Args:
        column: Column name, used for reporting.
        reference: Values of the baseline split.
        current: Values of the split being checked.
        psi_threshold: PSI above which the column counts as drifted.
        significance_level: KS p-value below which the column counts as drifted.

    Returns:
        A :class:`ColumnDrift` with the verdict and the supporting statistics.
    """
    reference = pd.Series(reference).dropna().to_numpy(dtype=float)
    current = pd.Series(current).dropna().to_numpy(dtype=float)

    if reference.size == 0 or current.size == 0:
        # Nothing to compare: treat as no drift rather than failing the run.
        return ColumnDrift(
            column=column, psi=0.0, ks_statistic=0.0, p_value=1.0, is_drifted=False
        )

    psi = population_stability_index(reference, current)
    ks_result = ks_2samp(reference, current)
    ks_statistic = float(ks_result.statistic)
    p_value = float(ks_result.pvalue)

    is_drifted = psi > psi_threshold or p_value < significance_level
    return ColumnDrift(
        column=column,
        psi=psi,
        ks_statistic=ks_statistic,
        p_value=p_value,
        is_drifted=is_drifted,
    )


def detect_drift(
    reference_df: DataFrame,
    current_df: DataFrame,
    psi_threshold: float = PSI_DRIFT_THRESHOLD,
    significance_level: float = KS_SIGNIFICANCE_LEVEL,
    max_drifted_ratio: float = MAX_DRIFTED_COLUMN_RATIO,
) -> DriftReport:
    """Compare every shared numeric column of two splits.

    Args:
        reference_df: Baseline split, normally the training data.
        current_df: Split to check against the baseline.
        psi_threshold: PSI above which a column counts as drifted.
        significance_level: KS p-value below which a column counts as drifted.
        max_drifted_ratio: Share of drifted columns above which the dataset is
            considered drifted.

    Returns:
        A :class:`DriftReport`.
    """
    shared_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[column])
        and pd.api.types.is_numeric_dtype(current_df[column])
    ]

    per_column = [
        detect_column_drift(
            column,
            reference_df[column],
            current_df[column],
            psi_threshold=psi_threshold,
            significance_level=significance_level,
        )
        for column in shared_columns
    ]

    drifted_columns = [column.column for column in per_column if column.is_drifted]
    n_drifted = len(drifted_columns)
    n_features = len(per_column)
    drifted_ratio = n_drifted / n_features if n_features else 0.0

    return DriftReport(
        n_features=n_features,
        n_drifted_features=n_drifted,
        drifted_columns=drifted_columns,
        dataset_drift=drifted_ratio > max_drifted_ratio,
        per_column=per_column,
    )
