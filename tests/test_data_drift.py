"""Tests for the drift detection used between ingestion and training."""

import numpy as np
import pandas as pd
import pytest

from src.components.data_drift import (
    PSI_DRIFT_THRESHOLD,
    detect_column_drift,
    detect_drift,
    population_stability_index,
)


def sample(size: int = 400, shift: float = 0.0, seed: int = 0) -> np.ndarray:
    """A normal sample, optionally shifted to simulate a distribution change."""
    generator = np.random.default_rng(seed)
    return generator.normal(loc=shift, scale=1.0, size=size)


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self):
        values = sample()

        psi = population_stability_index(values, values)

        assert psi < 0.01

    def test_a_large_shift_scores_above_the_threshold(self):
        psi = population_stability_index(sample(), sample(shift=4.0))

        assert psi > PSI_DRIFT_THRESHOLD

    def test_grows_with_the_size_of_the_shift(self):
        small_shift = population_stability_index(sample(), sample(shift=0.5))
        large_shift = population_stability_index(sample(), sample(shift=3.0))

        assert large_shift > small_shift

    def test_rejects_a_single_bucket(self):
        with pytest.raises(ValueError):
            population_stability_index(sample(), sample(), n_buckets=1)

    def test_handles_constant_columns(self):
        constant = np.full(100, 7.0)

        psi = population_stability_index(constant, constant)

        assert np.isfinite(psi)


class TestDetectColumnDrift:
    def test_reports_no_drift_for_the_same_data(self):
        values = sample()

        result = detect_column_drift("Income", values, values)

        assert result.is_drifted is False
        assert result.psi < PSI_DRIFT_THRESHOLD
        assert result.p_value > 0.05

    def test_flags_a_shifted_column(self):
        result = detect_column_drift("Income", sample(), sample(shift=3.0))

        assert result.is_drifted is True
        assert result.ks_statistic > 0.5

    def test_ignores_missing_values(self):
        reference = sample()
        current = sample(shift=3.0)
        current[0:10] = np.nan

        result = detect_column_drift("Income", reference, current)

        assert result.is_drifted is True

    def test_treats_an_empty_column_as_no_drift(self):
        result = detect_column_drift("Empty", pd.Series([np.nan] * 10), sample())

        assert result.is_drifted is False

    def test_serialises_to_a_plain_dict(self):
        result = detect_column_drift("Income", sample(), sample(shift=3.0))

        payload = result.to_dict()

        assert set(payload) == {"column", "psi", "ks_statistic", "p_value", "is_drifted"}


class TestDetectDrift:
    #: Row count of the frames built by :meth:`_frame`.
    ROWS = 200

    def _frame(self, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Age": sample(size=self.ROWS, seed=seed, shift=shift),
                "Income": sample(size=self.ROWS, seed=seed + 1, shift=shift),
                "Education": ["Basic", "Master"] * (self.ROWS // 2),
            }
        )

    def test_reports_no_drift_for_two_samples_of_the_same_source(self):
        report = detect_drift(self._frame(seed=0), self._frame(seed=2))

        assert report.n_features == 2  # the non numeric column is skipped
        assert report.dataset_drift is False
        assert report.drifted_columns == []

    def test_flags_dataset_drift_when_most_columns_move(self):
        report = detect_drift(self._frame(), self._frame(shift=5.0))

        assert report.n_drifted_features == 2
        assert report.dataset_drift is True
        assert set(report.drifted_columns) == {"Age", "Income"}

    def test_only_compares_shared_numeric_columns(self):
        reference = self._frame()
        current = self._frame().drop(columns=["Age"])

        report = detect_drift(reference, current)

        assert report.n_features == 1

    def test_report_serialises_to_a_plain_dict(self):
        report = detect_drift(self._frame(), self._frame(shift=5.0))

        payload = report.to_dict()

        assert payload["n_features"] == 2
        assert len(payload["columns"]) == 2
        assert payload["dataset_drift"] is True
