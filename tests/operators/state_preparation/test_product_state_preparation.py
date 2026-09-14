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

"""Tests for unentangled product-state preparation."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import ProductStatePreparation


def test_prepares_complex_product_state_in_qubit_order_and_copies_input() -> None:
    q0 = np.array([1j / np.sqrt(2), -1 / np.sqrt(2)], dtype=complex)
    q1 = np.array([0.0, np.exp(0.37j)], dtype=complex)
    states = [q0.tolist(), q1.tolist()]
    preparation = ProductStatePreparation(states)
    states[0][0] = 1.0
    expected = np.kron(q1, q0)

    preparation.validate()
    assert preparation.num_qubits() == 2
    assert preparation.num_parameters() == 0
    np.testing.assert_allclose(
        preparation.build_circuit("unused").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-14,
    )


def test_corrects_accepted_norm_roundoff() -> None:
    preparation = ProductStatePreparation([[1.0 + 5e-13, 0.0]])
    np.testing.assert_allclose(
        preparation.build_circuit("").to_matrix()[:, 0],
        [1.0, 0.0],
        rtol=0.0,
        atol=1e-14,
    )


def test_large_finite_modulus_is_a_value_error_not_an_overflow_error() -> None:
    value = complex(1.3e308, 1.3e308)
    with pytest.raises(ValueError, match="unit norm"):
        ProductStatePreparation([[value, 0.0]])


@pytest.mark.parametrize(
    "states,error",
    [
        ([], ValueError),
        ("bad", TypeError),
        ([[1.0]], ValueError),
        ([[1.0, 1.0]], ValueError),
        ([[0.0, 0.0]], ValueError),
        ([[float("nan"), 0.0]], ValueError),
        ([[object(), 0.0]], TypeError),
    ],
)
def test_rejects_invalid_product_factors(
    states: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        ProductStatePreparation(states)  # type: ignore[arg-type]
