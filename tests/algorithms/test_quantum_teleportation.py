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

"""Independent reference, integration, and boundary tests for teleportation."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import ClassicalExpr, ClassicalType

from cqlib_algorithms.algorithms import (
    QuantumTeleportation,
    QuantumTeleportationRequest,
    QuantumTeleportationResult,
)
from cqlib_algorithms.operators.state_preparation import UserCircuitPreparation

_I = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)


def _single(matrix: np.ndarray, qubit: int) -> np.ndarray:
    factors = [matrix if index == qubit else _I for index in (2, 1, 0)]
    return np.kron(np.kron(factors[0], factors[1]), factors[2])


def _cx(control: int, target: int) -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=complex)
    for source in range(8):
        destination = source
        if (source >> control) & 1:
            destination ^= 1 << target
        matrix[destination, source] = 1.0
    return matrix


def _reference_premeasurement(input_matrix: np.ndarray) -> np.ndarray:
    zero = np.zeros(8, dtype=complex)
    zero[0] = 1.0
    return (
        _single(_H, 0)
        @ _cx(0, 1)
        @ _cx(1, 2)
        @ _single(_H, 1)
        @ _single(input_matrix, 0)
        @ zero
    )


def _branch_bob_state(
    state: np.ndarray,
    phase_bit: int,
    bit_flip_bit: int,
) -> tuple[float, np.ndarray]:
    indices = [phase_bit + 2 * bit_flip_bit + 4 * bob for bob in (0, 1)]
    branch = state[indices]
    probability = float(np.vdot(branch, branch).real)
    branch = branch / math.sqrt(probability)
    if bit_flip_bit:
        branch = _X @ branch
    if phase_bit:
        branch = _Z @ branch
    return probability, branch


def _density(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def _ry(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta / 2.0), -math.sin(theta / 2.0)],
            [math.sin(theta / 2.0), math.cos(theta / 2.0)],
        ],
        dtype=complex,
    )


def _rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def _case_circuit(theta: float, phi: float, global_phase: float = 0.0) -> Circuit:
    circuit = Circuit([Qubit(7)])
    circuit.ry(Qubit(7), theta)
    circuit.rz(Qubit(7), phi)
    circuit.set_global_phase(global_phase)
    return circuit


@pytest.mark.parametrize(
    ("theta", "phi", "global_phase"),
    [
        (0.0, 0.0, 0.0),
        (math.pi, 0.0, 0.0),
        (math.pi / 2.0, 0.0, 0.0),
        (math.pi / 2.0, math.pi, 0.0),
        (math.pi / 2.0, math.pi / 2.0, 0.0),
        (math.pi / 2.0, -math.pi / 2.0, 0.0),
        (0.73, -1.19, 0.0),
        (2.21, 0.42, -0.37),
    ],
)
def test_all_bell_branches_match_independent_numpy_reference(
    theta: float,
    phi: float,
    global_phase: float,
) -> None:
    input_matrix = np.exp(1j * global_phase) * _rz(phi) @ _ry(theta)
    target = input_matrix[:, 0]
    expected = _reference_premeasurement(input_matrix)
    actual = QuantumTeleportation().build_premeasurement_circuit(
        UserCircuitPreparation(_case_circuit(theta, phi, global_phase))
    )

    assert actual.qubits == [Qubit(0), Qubit(1), Qubit(2)]
    assert actual.to_matrix()[:, 0] == pytest.approx(expected, abs=1e-12)
    for phase_bit, bit_flip_bit in ((0, 0), (0, 1), (1, 0), (1, 1)):
        probability, restored = _branch_bob_state(expected, phase_bit, bit_flip_bit)
        assert probability == pytest.approx(0.25, abs=1e-12)
        assert _density(restored) == pytest.approx(_density(target), abs=1e-12)


def test_real_core_measurements_and_corrections_restore_general_complex_state() -> None:
    request = QuantumTeleportationRequest(
        UserCircuitPreparation(_case_circuit(1.137, -0.843, 0.29)),
        shots=256,
    )

    result = QuantumTeleportation().run([request])[0]

    assert result.shots == 256
    assert sum(result.bell_counts) == 256
    assert sum(result.bell_probabilities) == pytest.approx(1.0)
    assert result.minimum_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.mean_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.successful_shots == 256
    assert result.success_rate == 1.0
    assert result.all_succeeded


def test_batch_and_empty_batch_preserve_request_order() -> None:
    requests = [
        QuantumTeleportationRequest(UserCircuitPreparation(_case_circuit(0.2, 0.7)), 3),
        QuantumTeleportationRequest(
            UserCircuitPreparation(_case_circuit(2.0, -0.3)), 5
        ),
    ]

    results = QuantumTeleportation().run(requests)

    assert [result.shots for result in results] == [3, 5]
    assert all(result.all_succeeded for result in results)
    assert QuantumTeleportation().run([]) == []


def test_source_circuit_is_not_modified_and_build_returns_fresh_circuits() -> None:
    source = _case_circuit(0.42, 0.91, -0.13)
    before = source.to_matrix([7]).copy()
    before_qubits = list(source.qubits)
    algorithm = QuantumTeleportation()

    first = algorithm.build_premeasurement_circuit(UserCircuitPreparation(source))
    second = algorithm.build_premeasurement_circuit(UserCircuitPreparation(source))
    first.x(2)
    algorithm.run([QuantumTeleportationRequest(UserCircuitPreparation(source), 2)])

    assert first is not second
    assert len(first.operations) == len(second.operations) + 1
    assert source.qubits == before_qubits
    assert source.to_matrix([7]) == pytest.approx(before, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_preparation": object(), "shots": 1},
        {"state_preparation": Circuit(1), "shots": 1},
        {"state_preparation": UserCircuitPreparation(Circuit(2)), "shots": 1},
        {"state_preparation": UserCircuitPreparation(Circuit(1)), "shots": True},
        {"state_preparation": UserCircuitPreparation(Circuit(1)), "shots": 0},
        {
            "state_preparation": UserCircuitPreparation(Circuit(1)),
            "shots": 1,
            "fidelity_tolerance": True,
        },
        {
            "state_preparation": UserCircuitPreparation(Circuit(1)),
            "shots": 1,
            "fidelity_tolerance": -1.0,
        },
        {
            "state_preparation": UserCircuitPreparation(Circuit(1)),
            "shots": 1,
            "fidelity_tolerance": 1.1,
        },
        {
            "state_preparation": UserCircuitPreparation(Circuit(1)),
            "shots": 1,
            "fidelity_tolerance": math.nan,
        },
    ],
)
def test_request_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        QuantumTeleportationRequest(**kwargs)  # type: ignore[arg-type]


def test_request_rejects_active_symbols() -> None:
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))

    with pytest.raises(ValueError, match="active symbols"):
        QuantumTeleportationRequest(UserCircuitPreparation(circuit), 1)


@pytest.mark.parametrize("directive", ["measure", "reset", "delay"])
def test_request_rejects_direct_dynamic_operations(directive: str) -> None:
    request = QuantumTeleportationRequest(UserCircuitPreparation(Circuit(1)), 1)
    circuit = request.preparation_circuit
    assert circuit is not None
    if directive == "measure":
        circuit.measure(0)
    elif directive == "reset":
        circuit.reset(0)
    else:
        circuit.delay(0, 1.0)

    with pytest.raises(ValueError, match="unitary|invertible"):
        request.validate()


def test_request_rejects_nested_dynamic_operations() -> None:
    inner = Circuit(1)
    inner.reset(0)
    request = QuantumTeleportationRequest(UserCircuitPreparation(Circuit(1)), 1)
    outer = request.preparation_circuit
    assert outer is not None
    outer.append_circuit_gate(inner.to_gate("hidden_reset"), [0])

    with pytest.raises(ValueError, match="unitary"):
        request.validate()


@pytest.mark.parametrize("operation", ["classical_data", "classical_control"])
def test_request_rejects_classical_data_and_control(operation: str) -> None:
    request = QuantumTeleportationRequest(UserCircuitPreparation(Circuit(1)), 1)
    circuit = request.preparation_circuit
    assert circuit is not None
    variable = circuit.var(ClassicalType.bool())
    if operation == "classical_data":
        circuit.store(variable, ClassicalExpr.bool_literal(True))
    else:
        circuit.if_(variable.expr(), lambda body: body.x(0))

    with pytest.raises(ValueError, match="unitary"):
        request.validate()


def test_mutated_shallow_request_is_revalidated_before_any_measurement() -> None:
    valid = Circuit(1)
    first = QuantumTeleportationRequest(UserCircuitPreparation(valid), 8)
    mutable = Circuit(1)
    second = QuantumTeleportationRequest(UserCircuitPreparation(mutable), 8)
    second.preparation_circuit.ry(0, Parameter("late"))

    with pytest.raises(ValueError, match="active symbols"):
        QuantumTeleportation().run([first, second])


def test_request_and_result_are_frozen_and_replaceable() -> None:
    request = QuantumTeleportationRequest(UserCircuitPreparation(Circuit(1)), 2)
    result = QuantumTeleportationResult(((0, 0), (1, 1)), (1.0, 1.0))

    with pytest.raises(FrozenInstanceError):
        request.shots = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.fidelity_tolerance = 0.1  # type: ignore[misc]
    assert replace(request, shots=4).shots == 4
    assert replace(result, fidelity_tolerance=0.1).fidelity_tolerance == 0.1


def test_result_uses_fixed_bell_order_and_fidelity_threshold() -> None:
    result = QuantumTeleportationResult(
        ((1, 1), (0, 0), (0, 1), (1, 1)),
        (1.0, 1.0 - 5e-4, 0.98, 1.0),
        1e-3,
    )

    assert result.bell_counts == (1, 1, 0, 2)
    assert result.bell_probabilities == (0.25, 0.25, 0.0, 0.5)
    assert result.minimum_fidelity == 0.98
    assert result.mean_fidelity == pytest.approx(0.994875)
    assert result.successful_shots == 3
    assert result.success_rate == 0.75
    assert not result.all_succeeded


@pytest.mark.parametrize(
    ("measurements", "fidelities", "tolerance"),
    [
        ([], (1.0,), 1e-12),
        ((), (), 1e-12),
        (((0, 0),), (), 1e-12),
        (((0,),), (1.0,), 1e-12),
        (((0, 2),), (1.0,), 1e-12),
        (((False, 0),), (1.0,), 1e-12),
        (((0, 0),), [1.0], 1e-12),
        (((0, 0),), (math.nan,), 1e-12),
        (((0, 0),), (-1e-6,), 1e-12),
        (((0, 0),), (1.0 + 1e-6,), 1e-12),
        (((0, 0),), (1.0,), -1.0),
    ],
)
def test_result_rejects_invalid_values(
    measurements: object,
    fidelities: object,
    tolerance: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        QuantumTeleportationResult(  # type: ignore[arg-type]
            measurements, fidelities, tolerance
        )


def test_result_clamps_only_roundoff_sized_fidelity_excursions() -> None:
    result = QuantumTeleportationResult(
        ((0, 0), (0, 1)),
        (-5e-13, 1.0 + 5e-13),
    )

    assert result.fidelities == (0.0, 1.0)


def test_run_rejects_non_request_entries() -> None:
    with pytest.raises(TypeError, match="QuantumTeleportationRequest"):
        QuantumTeleportation().run([object()])  # type: ignore[list-item]


def test_single_run_accepts_complex_preparation_component():
    from cqlib_algorithms.operators import ArbitraryStatePreparation

    preparation = ArbitraryStatePreparation([1 / math.sqrt(3), 1j * math.sqrt(2 / 3)])
    algorithm = QuantumTeleportation()
    result = algorithm.run(preparation, shots=16)
    assert result.shots == 16
    np.testing.assert_allclose(result.fidelities, np.ones(16), atol=1e-12)


@pytest.mark.parametrize(
    "case", ["width", "symbol", "missing_shots", "shots", "tolerance", "batch"]
)
def test_single_teleportation_input_boundaries(case):
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import AngleEncoding, GHZStatePreparation

    algorithm = QuantumTeleportation()
    preparation = GHZStatePreparation(1)
    with pytest.raises((TypeError, ValueError)):
        if case == "width":
            algorithm.run(GHZStatePreparation(2), shots=4)
        elif case == "symbol":
            algorithm.run(AngleEncoding(1, StandardGate.RY), shots=4)
        elif case == "missing_shots":
            algorithm.run(preparation)
        elif case == "shots":
            algorithm.run(preparation, shots=0)
        elif case == "tolerance":
            algorithm.run(preparation, shots=4, fidelity_tolerance=math.nan)
        else:
            algorithm.run([], shots=4)


def test_component_requests_preserve_mixed_batch_order_and_complex_states():
    from cqlib_algorithms.operators import (
        ArbitraryStatePreparation,
        BasisStateInitialization,
    )

    circuit = _case_circuit(0.73, -1.19, 0.37)
    requests = [
        QuantumTeleportationRequest(
            ArbitraryStatePreparation([math.sqrt(0.3), 1j * math.sqrt(0.7)]), 3
        ),
        QuantumTeleportationRequest(BasisStateInitialization(1, 1), 5),
        QuantumTeleportationRequest(UserCircuitPreparation(circuit), 7),
    ]

    results = QuantumTeleportation().run(requests)
    assert [item.shots for item in results] == [3, 5, 7]
    for result in results:
        assert result.fidelities == pytest.approx([1.0] * result.shots, abs=1e-12)
    assert requests[2].preparation_circuit is not circuit
    np.testing.assert_allclose(
        requests[2].preparation_circuit.to_matrix(), circuit.to_matrix(), atol=1e-12
    )


def test_component_request_snapshots_once_and_preserves_phase_and_qubit_order(
    monkeypatch: pytest.MonkeyPatch,
):
    from cqlib_algorithms.operators import GHZStatePreparation

    source = _case_circuit(0.73, -1.19, 0.37)
    expected = source.to_matrix().copy()
    prefixes: list[str] = []
    original_build = GHZStatePreparation.build_circuit

    def build(_self: GHZStatePreparation, prefix: str) -> Circuit:
        if _self.num_qubits() != 1:
            return original_build(_self, prefix)
        prefixes.append(prefix)
        return source

    monkeypatch.setattr(GHZStatePreparation, "build_circuit", build)
    request = QuantumTeleportationRequest(GHZStatePreparation(1), shots=4)
    assert isinstance(request.preparation_circuit, Circuit)
    assert request.preparation_circuit is not source
    assert request.preparation_circuit.qubits == [Qubit(0)]
    assert request.preparation_circuit.to_matrix() == pytest.approx(expected, abs=1e-12)
    source.reset(Qubit(7))

    algorithm = QuantumTeleportation()
    expected_state = _reference_premeasurement(expected)
    actual = algorithm.build_premeasurement_circuit(request.state_preparation)
    assert actual.to_matrix()[:, 0] == pytest.approx(expected_state, abs=1e-12)
    for _ in range(2):
        result = algorithm.run([request])[0]
        assert result.fidelities == pytest.approx([1.0] * 4, abs=1e-12)
    assert prefixes == ["teleportation_input"]
    copied = replace(request, shots=6)
    assert copied.preparation_circuit is not request.preparation_circuit
    np.testing.assert_allclose(
        copied.preparation_circuit.to_matrix(), expected, atol=1e-12
    )
    assert prefixes == ["teleportation_input"]


@pytest.mark.parametrize("case", ["width", "symbol"])
def test_component_request_rejects_incompatible_preparations(case: str):
    from cqlib.circuit.gates import StandardGate

    from cqlib_algorithms.operators import AngleEncoding, GHZStatePreparation

    preparation = (
        GHZStatePreparation(2) if case == "width" else AngleEncoding(1, StandardGate.RY)
    )
    with pytest.raises(ValueError, match="one qubit|active symbols"):
        QuantumTeleportationRequest(preparation, 1)


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("type", TypeError, "return a Circuit"),
        ("width", ValueError, "qubit count"),
        ("reset", ValueError, "unitary|invertible"),
    ],
)
def test_component_request_checks_built_circuit(
    case: str, error: type[Exception], message: str, monkeypatch: pytest.MonkeyPatch
):
    from cqlib_algorithms.operators import GHZStatePreparation

    def build(_self: GHZStatePreparation, _prefix: str) -> Circuit:
        if case == "type":
            return object()  # type: ignore[return-value]
        circuit = Circuit(2 if case == "width" else 1)
        if case == "reset":
            circuit.reset(0)
        return circuit

    monkeypatch.setattr(GHZStatePreparation, "build_circuit", build)
    with pytest.raises(error, match=message):
        QuantumTeleportationRequest(GHZStatePreparation(1), 1)
