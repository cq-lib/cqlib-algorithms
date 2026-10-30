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

"""Tests for the quasi-probability decompositions."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.circuit import Instruction, StandardGate
from cqlib.device import NoiseModel, SingleQubitNoise, TwoQubitNoise

from cqlib import Circuit
from cqlib_algorithms.error_mitigation.quasiprobability import (
    QuasiprobabilityRepresentation,
    depolarizing_quasiprobability_of_circuit,
    depolarizing_quasiprobability_of_operation,
    quasiprobability_of_circuit,
    quasiprobability_of_operation,
)

_NOISE_LEVEL = 0.03
_ATOL = 1e-12


def _hadamard() -> Instruction:
    """Return the identity-free one-qubit instruction used across the tests."""
    return Instruction.from_standard_gate(StandardGate.H)


def _cx() -> Instruction:
    """Return the two-qubit instruction used across the tests."""
    return Instruction.from_standard_gate(StandardGate.CX)


def _one_qubit_weights(noise_level: float) -> list[float]:
    """Return the analytic one-qubit weights of the depolarizing expansion."""
    epsilon = 4.0 / 3.0 * noise_level
    ratio = epsilon / (1.0 - epsilon)
    return [1.0 + 0.75 * ratio] + 3 * [-0.25 * ratio]


def _two_qubit_weights(noise_level: float) -> list[float]:
    """Return the analytic two-qubit weights of the depolarizing expansion."""
    epsilon = 16.0 / 15.0 * noise_level
    ratio = epsilon / (1.0 - epsilon)
    return [1.0 + (15.0 / 16.0) * ratio] + 15 * [-(1.0 / 16.0) * ratio]


def test_one_qubit_basis_is_the_identity_plus_three_paulis() -> None:
    expansion = depolarizing_quasiprobability_of_operation(
        _hadamard(),
        _NOISE_LEVEL,
    )
    assert len(expansion.basis) == 4
    assert len(expansion.coefficients) == 4
    identity = expansion.basis[0].standard_gate.matrix()
    assert np.allclose(identity, np.eye(2))


def test_one_qubit_pauli_corrections_are_x_y_z_in_order() -> None:
    expansion = depolarizing_quasiprobability_of_operation(
        _hadamard(),
        _NOISE_LEVEL,
    )
    expected = {
        1: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        2: np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        3: np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    for index, matrix in expected.items():
        assert np.allclose(expansion.basis[index].standard_gate.matrix(), matrix)


def test_one_qubit_closed_form_matches_the_analytic_weights() -> None:
    expansion = depolarizing_quasiprobability_of_operation(
        _hadamard(),
        _NOISE_LEVEL,
    )
    assert np.allclose(expansion.coefficients, _one_qubit_weights(_NOISE_LEVEL))
    epsilon = 4.0 / 3.0 * _NOISE_LEVEL
    assert expansion.norm == pytest.approx(1.0 + 1.5 * epsilon / (1.0 - epsilon))


def test_one_qubit_closed_form_matches_the_matrix_path() -> None:
    closed = depolarizing_quasiprobability_of_operation(
        _hadamard(),
        _NOISE_LEVEL,
    )
    matrix = quasiprobability_of_operation(
        _hadamard(),
        SingleQubitNoise.depolarizing(_NOISE_LEVEL),
    )
    assert np.allclose(closed.coefficients, matrix.coefficients, atol=_ATOL)


def test_two_qubit_basis_has_sixteen_entries() -> None:
    expansion = depolarizing_quasiprobability_of_operation(_cx(), _NOISE_LEVEL)
    assert len(expansion.basis) == 16
    assert len(expansion.coefficients) == 16


def test_two_qubit_closed_form_matches_the_analytic_weights() -> None:
    expansion = depolarizing_quasiprobability_of_operation(_cx(), _NOISE_LEVEL)
    assert np.allclose(expansion.coefficients, _two_qubit_weights(_NOISE_LEVEL))


def test_two_qubit_closed_form_matches_the_matrix_path() -> None:
    closed = depolarizing_quasiprobability_of_operation(_cx(), _NOISE_LEVEL)
    matrix = quasiprobability_of_operation(
        _cx(),
        TwoQubitNoise.depolarizing(_NOISE_LEVEL),
    )
    assert np.allclose(closed.coefficients, matrix.coefficients, atol=_ATOL)


def test_depolarizing_expansion_of_circuit_skips_measurements() -> None:
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0)
    circuit.measure(1)
    expansions = depolarizing_quasiprobability_of_circuit(circuit, _NOISE_LEVEL)
    assert len(expansions) == 2
    assert len(expansions[0].basis) == 4
    assert len(expansions[1].basis) == 16


def test_matrix_path_expands_a_circuit_against_a_noise_model() -> None:
    model = NoiseModel()
    model.add_single_qubit_error(
        StandardGate.H,
        0,
        SingleQubitNoise.depolarizing(_NOISE_LEVEL),
    )
    model.add_two_qubit_error(
        StandardGate.CX,
        0,
        1,
        TwoQubitNoise.depolarizing(_NOISE_LEVEL),
    )
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    expansions = quasiprobability_of_circuit(circuit, model)
    closed = depolarizing_quasiprobability_of_circuit(circuit, _NOISE_LEVEL)
    assert len(expansions) == 2
    assert np.allclose(expansions[0].coefficients, closed[0].coefficients, atol=_ATOL)
    assert np.allclose(expansions[1].coefficients, closed[1].coefficients, atol=_ATOL)


def test_matrix_path_rejects_a_circuit_without_a_registered_channel() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    with pytest.raises(ValueError):
        quasiprobability_of_circuit(circuit, NoiseModel())


def test_operation_rejects_an_unknown_noise_type() -> None:
    with pytest.raises(TypeError):
        quasiprobability_of_operation(_hadamard(), _NOISE_LEVEL)


def test_operation_rejects_a_mismatched_qubit_count() -> None:
    with pytest.raises(ValueError):
        quasiprobability_of_operation(
            _cx(),
            SingleQubitNoise.depolarizing(_NOISE_LEVEL),
        )


def test_operation_rejects_a_three_qubit_gate() -> None:
    gate = Instruction.from_standard_gate(StandardGate.CCX)
    with pytest.raises(ValueError):
        depolarizing_quasiprobability_of_operation(gate, _NOISE_LEVEL)


def test_operation_rejects_a_divergent_one_qubit_noise_level() -> None:
    with pytest.raises(ValueError):
        depolarizing_quasiprobability_of_operation(_hadamard(), 0.9)


def test_operation_rejects_a_divergent_two_qubit_noise_level() -> None:
    with pytest.raises(ValueError):
        depolarizing_quasiprobability_of_operation(_cx(), 0.95)


def test_operation_rejects_a_negative_noise_level() -> None:
    with pytest.raises(ValueError):
        depolarizing_quasiprobability_of_operation(_hadamard(), -0.1)


def test_representation_norm_is_the_weight_one_norm() -> None:
    expansion = QuasiprobabilityRepresentation(
        _hadamard(),
        (_hadamard(), _hadamard()),
        (0.5, -1.5),
    )
    assert expansion.norm == pytest.approx(2.0)


def test_representation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        QuasiprobabilityRepresentation(
            _hadamard(),
            (_hadamard(),),
            (1.0, 2.0),
        )
