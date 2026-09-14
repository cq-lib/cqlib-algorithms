"""Tests for normalized dense amplitude encoding."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import AmplitudeEncoding


@pytest.mark.reference
def test_normalizes_copies_and_encodes_complex_features_exactly() -> None:
    features = [1.0 + 2.0j, -3.0j, 4.0, -2.0 + 0.5j]
    expected = np.asarray(features, dtype=complex)
    expected /= np.linalg.norm(expected)
    encoding = AmplitudeEncoding(features)
    features[:] = [1.0, 0.0, 0.0, 0.0]

    encoding.validate()
    assert encoding.num_qubits() == 2
    assert encoding.num_parameters() == 0
    np.testing.assert_allclose(
        encoding.build_circuit("unused").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-13,
    )


def test_normalizes_large_finite_values_without_overflow() -> None:
    encoding = AmplitudeEncoding([1e308, 1e308j])
    expected = np.array([1.0, 1j]) / np.sqrt(2)
    np.testing.assert_allclose(
        encoding.build_circuit("").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-13,
    )


def test_normalizes_finite_complex_values_whose_unscaled_modulus_overflows() -> None:
    value = complex(1.3e308, 1.3e308)
    encoding = AmplitudeEncoding([value, value.conjugate()])
    expected = np.array([0.5 + 0.5j, 0.5 - 0.5j])
    np.testing.assert_allclose(
        encoding.build_circuit("").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "features,error",
    [
        ([], ValueError),
        ([1.0], ValueError),
        ([1.0, 0.0, 0.0], ValueError),
        ([0.0, 0.0], ValueError),
        ([complex(float("nan"), 0.0), 1.0], ValueError),
        ([object(), 0.0], TypeError),
        (None, TypeError),
    ],
)
def test_rejects_invalid_features(features: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        AmplitudeEncoding(features)  # type: ignore[arg-type]
