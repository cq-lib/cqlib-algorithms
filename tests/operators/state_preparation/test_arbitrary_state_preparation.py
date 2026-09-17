# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for exact arbitrary pure-state preparation."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import ArbitraryStatePreparation


@pytest.mark.reference
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
def test_random_complex_states_match_every_amplitude(num_qubits: int) -> None:
    rng = np.random.default_rng(4100 + num_qubits)
    for _ in range(4):
        expected = rng.normal(size=1 << num_qubits) + 1j * rng.normal(
            size=1 << num_qubits
        )
        expected /= np.linalg.norm(expected)
        preparation = ArbitraryStatePreparation(expected.tolist())
        actual = preparation.build_circuit("unused").to_matrix()[:, 0]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "expected",
    [
        np.array([np.exp(0.731j), 0.0]),
        np.array([0.0, -1j]),
        np.array([0.5j, -0.5, 0.5, -0.5j]),
    ],
)
def test_preserves_global_and_relative_phase(expected: np.ndarray) -> None:
    preparation = ArbitraryStatePreparation(expected.tolist())
    np.testing.assert_allclose(
        preparation.build_circuit("").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-13,
    )


def test_preserves_representable_extremely_small_complex_amplitude() -> None:
    tiny = 1e-200 * np.exp(0.619j)
    expected = np.array([tiny, np.sqrt(1.0 - abs(tiny) ** 2)], dtype=complex)
    preparation = ArbitraryStatePreparation(expected.tolist())
    actual = preparation.build_circuit("").to_matrix()[:, 0]
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=0.0)


def test_copies_input_and_corrects_accepted_norm_roundoff() -> None:
    values = [1.0 + 5e-13, 0.0]
    preparation = ArbitraryStatePreparation(values)
    values[0] = 0.0
    np.testing.assert_allclose(
        preparation.build_circuit("x").to_matrix()[:, 0],
        [1.0, 0.0],
        rtol=0.0,
        atol=1e-14,
    )


def test_large_finite_modulus_is_a_value_error_not_an_overflow_error() -> None:
    value = complex(1.3e308, 1.3e308)
    with pytest.raises(ValueError, match="unit norm"):
        ArbitraryStatePreparation([value, 0.0])


@pytest.mark.parametrize(
    "amplitudes,error",
    [
        ([1.0], ValueError),
        ([1.0, 0.0, 0.0], ValueError),
        ([0.0, 0.0], ValueError),
        ([1.0, 1.0], ValueError),
        ([float("nan"), 0.0], ValueError),
        ([float("inf"), 0.0], ValueError),
        ([object(), 0.0], TypeError),
        ("bad", TypeError),
    ],
)
def test_rejects_invalid_dense_states(
    amplitudes: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        ArbitraryStatePreparation(amplitudes)  # type: ignore[arg-type]


def test_returns_independent_circuits_and_validates_prefix() -> None:
    preparation = ArbitraryStatePreparation([1.0, 0.0])
    first = preparation.build_circuit("a")
    first.x(0)
    assert len(preparation.build_circuit("a").operations) == 0
    with pytest.raises(TypeError, match="prefix must be a str"):
        preparation.build_circuit(None)  # type: ignore[arg-type]
