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

"""Tests for the excitation-preserving ansatz."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.qis import PauliString

from cqlib_algorithms.operators.ansatz import ExcitationPreservingAnsatz
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


def _evolution(matrix: np.ndarray, time: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.exp(-1j * time * values)) @ vectors.conj().T


def _exchange(theta: float) -> np.ndarray:
    xx = PauliString.from_str("XX").to_matrix()
    yy = PauliString.from_str("YY").to_matrix()
    return _evolution(yy, theta / 4) @ _evolution(xx, theta / 4)


def test_iswap_exchange_block_matches_independent_matrix_reference():
    component = ExcitationPreservingAnsatz(
        2,
        reps=1,
        mode="iswap",
        skip_final_rotation_layer=True,
    )
    theta = 0.73
    actual = (
        component.build_circuit("p")
        .assign_parameters({"p_0": 0.0, "p_1": 0.0, "p_2": theta})
        .to_matrix()
    )

    assert component.num_parameters() == 3
    np.testing.assert_allclose(actual, _exchange(theta), rtol=0.0, atol=1e-12)


def test_fsim_mode_adds_controlled_phase_with_independent_parameter():
    component = ExcitationPreservingAnsatz(
        2,
        reps=1,
        mode="fsim",
        skip_final_rotation_layer=True,
    )
    theta = -0.52
    phi = 0.37
    actual = (
        component.build_circuit("p")
        .assign_parameters({"p_0": 0.0, "p_1": 0.0, "p_2": theta, "p_3": phi})
        .to_matrix()
    )
    controlled_phase = np.diag([1.0, 1.0, 1.0, np.exp(-1j * phi)])

    assert component.num_parameters() == 4
    np.testing.assert_allclose(
        actual,
        controlled_phase @ _exchange(theta),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ["iswap", "fsim"])
def test_circuit_commutes_with_total_excitation_number(mode):
    component = ExcitationPreservingAnsatz(3, reps=2, mode=mode)
    circuit = component.build_circuit("p")
    bindings = {
        name: 0.11 + 0.07 * index for index, name in enumerate(circuit.used_symbols)
    }
    unitary = circuit.assign_parameters(bindings).to_matrix()
    number = np.diag([index.bit_count() for index in range(8)])

    np.testing.assert_allclose(
        unitary @ number - number @ unitary,
        np.zeros((8, 8)),
        rtol=0.0,
        atol=1e-12,
    )


def test_each_hamming_weight_sector_is_preserved():
    component = ExcitationPreservingAnsatz(3, reps=2, mode="fsim")
    circuit = component.build_circuit("p")
    unitary = circuit.assign_parameters(
        {name: -0.2 + 0.03 * i for i, name in enumerate(circuit.used_symbols)}
    ).to_matrix()

    for column in range(8):
        weight = column.bit_count()
        forbidden = [row for row in range(8) if row.bit_count() != weight]
        np.testing.assert_allclose(
            unitary[forbidden, column],
            np.zeros(len(forbidden)),
            rtol=0.0,
            atol=1e-12,
        )


def test_custom_topology_and_unentangled_qubit_skipping_control_parameters():
    topology = EntanglementTopology.custom([(0, 2)])
    iswap = ExcitationPreservingAnsatz(
        3,
        reps=2,
        entanglement=topology,
        skip_unentangled_qubits=True,
    )
    fsim = ExcitationPreservingAnsatz(
        3,
        reps=2,
        mode="fsim",
        entanglement=topology,
        skip_unentangled_qubits=True,
    )

    assert iswap.num_parameters() == 8
    assert fsim.num_parameters() == 10
    assert [
        (operation.name, [qubit.index for qubit in operation.qubits])
        for operation in iswap.build_circuit("p").operations
    ] == [
        ("RZ", [0]),
        ("RZ", [2]),
        ("RXX", [0, 2]),
        ("RYY", [0, 2]),
        ("RZ", [0]),
        ("RZ", [2]),
        ("RXX", [0, 2]),
        ("RYY", [0, 2]),
        ("RZ", [0]),
        ("RZ", [2]),
    ]


def test_single_qubit_and_zero_repetition_degenerate_cases():
    active = ExcitationPreservingAnsatz(1, reps=0)
    skipped = ExcitationPreservingAnsatz(
        1,
        reps=0,
        skip_unentangled_qubits=True,
    )

    assert active.num_parameters() == 1
    assert active.build_circuit("p").used_symbols == ["p_0"]
    assert skipped.num_parameters() == 0
    np.testing.assert_array_equal(skipped.build_circuit("p").to_matrix(), np.eye(2))


def test_parameter_shift_matches_independent_finite_difference():
    ansatz = ExcitationPreservingAnsatz(
        2,
        reps=1,
        mode="iswap",
        skip_final_rotation_layer=True,
    ).build_circuit("p")
    circuit = Circuit(2)
    circuit.ry(0, 0.41)
    circuit.rx(1, -0.28)
    circuit.compose(ansatz)
    observable = PauliString.from_str("ZI")
    bindings = {"p_0": 0.13, "p_1": -0.27, "p_2": 0.62}
    names = tuple(circuit.used_symbols)
    request = GradientRequest(
        EstimatorRequest(circuit, observable, bindings),
        names,
    )

    actual = ParameterShiftGradient().run(StatevectorEstimator(), [request])[0].values

    def expectation(name: str, offset: float) -> float:
        shifted = dict(bindings)
        shifted[name] += offset
        state = circuit.assign_parameters(shifted).to_matrix()[:, 0]
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    step = 1e-6
    expected = tuple(
        (expectation(name, step) - expectation(name, -step)) / (2 * step)
        for name in names
    )
    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda: ExcitationPreservingAnsatz(2, mode="exchange"),
            ValueError,
            "iswap.*fsim",
        ),
        (
            lambda: ExcitationPreservingAnsatz(2, mode=1),
            TypeError,
            "mode must be a string",
        ),
        (
            lambda: ExcitationPreservingAnsatz(
                2, entanglement=EntanglementTopology.custom([(0, 0)])
            ),
            ValueError,
            "distinct qubits",
        ),
        (
            lambda: ExcitationPreservingAnsatz(
                2, entanglement=EntanglementTopology.custom([(0, 1), (1, 0)])
            ),
            ValueError,
            "duplicated",
        ),
        (
            lambda: ExcitationPreservingAnsatz(
                2, entanglement=EntanglementTopology.custom([(0, 2)])
            ),
            ValueError,
            "out of range",
        ),
    ],
)
def test_invalid_configuration_is_rejected(factory, error, message):
    with pytest.raises(error, match=message):
        factory()
