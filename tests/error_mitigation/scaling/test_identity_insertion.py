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

"""Tests for the identity-insertion noise-scaling primitive."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib import Circuit
from cqlib_algorithms.error_mitigation.scaling import insert_id_layers


def _body() -> Circuit:
    """Return the two-operation, two-qubit body used across the tests."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _measured(circuit: Circuit) -> Circuit:
    """Append one measurement per qubit and return the same circuit."""
    for qubit in range(len(circuit.qubits)):
        circuit.measure(qubit)
    return circuit


def _names(circuit: Circuit) -> list[str]:
    """Return the operation names of ``circuit``, in execution order."""
    return [operation.name for operation in circuit.operations]


def _identity_count(circuit: Circuit) -> int:
    """Return how many identity gates ``circuit`` carries."""
    return sum(1 for operation in circuit.operations if operation.name.upper() == "I")


def _measured_qubit_count(circuit: Circuit) -> int:
    """Return how many qubits ``circuit`` measures."""
    return sum(
        len(operation.qubits)
        for operation in circuit.operations
        if operation.is_classical_data
    )


def _phase(circuit: Circuit) -> complex:
    """Return the global phase of ``circuit`` as a complex scalar."""
    return complex(circuit.global_phase.evaluate({}))


@pytest.mark.parametrize(
    ("scale_factor", "expected_identities"),
    [(1.0, 0), (2.0, 4), (2.5, 6), (3.0, 8)],
)
def test_insert_id_layers_inserts_the_requested_depth(
    scale_factor: float,
    expected_identities: int,
) -> None:
    body = _body()
    scaled = insert_id_layers(body, scale_factor, seed=5)
    num_layers = math.floor(len(body.operations) * (scale_factor - 1.0))
    num_qubits = len(body.qubits)
    assert _identity_count(scaled) == expected_identities
    assert _identity_count(scaled) == num_layers * num_qubits
    assert len(scaled.operations) == len(body.operations) + expected_identities
    assert np.allclose(body.to_matrix(), scaled.to_matrix())


def test_insert_id_layers_at_scale_one_changes_nothing() -> None:
    body = _body()
    assert _names(insert_id_layers(body, 1.0, seed=5)) == _names(body)


def test_insert_id_layers_is_deterministic_in_the_seed() -> None:
    first = insert_id_layers(_body(), 2.4, seed=11)
    second = insert_id_layers(_body(), 2.4, seed=11)
    assert _names(first) == _names(second)


def test_insert_id_layers_preserves_the_global_phase() -> None:
    body = _body()
    body.set_global_phase(0.5)
    assert _phase(insert_id_layers(body, 3.0, seed=1)) == pytest.approx(0.5)


def test_insert_id_layers_strips_and_restores_measurements() -> None:
    measured = _measured(_body())
    scaled = insert_id_layers(measured, 2.0, seed=1)
    assert _measured_qubit_count(measured) == 2
    assert _measured_qubit_count(scaled) == 2


def test_insert_id_layers_rejects_a_scale_below_one() -> None:
    with pytest.raises(ValueError):
        insert_id_layers(_body(), 0.5, seed=1)
