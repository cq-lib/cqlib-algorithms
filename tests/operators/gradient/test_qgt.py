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

"""QGT tests against analytic values and finite-difference references."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.circuit import FrozenCircuit, SymbolicComplex, SymbolicMatrix
from cqlib.circuit.gates import CircuitGate, StandardGate, UnitaryGate
from cqlib.qis import PauliString

from cqlib import Circuit, Parameter, Qubit
from cqlib_algorithms.operators.gradient import QGTResult, StatevectorQGT
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


@pytest.fixture
def qgt() -> StatevectorQGT:
    return StatevectorQGT()


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


def _difference_tensor(circuit, bindings, names, step=1e-6) -> np.ndarray:
    """Independent QGT reference via central differences on circuit matrices."""

    def state(deltas: dict[str, float]) -> np.ndarray:
        shifted = {
            name: value + deltas.get(name, 0.0) for name, value in bindings.items()
        }
        bound = circuit.assign_parameters(shifted)
        qubit_order = [qubit.index for qubit in bound.qubits]
        return bound.to_matrix(qubit_order)[:, 0]

    psi = state({})
    columns = []
    for name in names:
        plus = state({name: step})
        minus = state({name: -step})
        columns.append((plus - minus) / (2 * step))
    count = len(names)
    tensor = np.zeros((count, count), dtype=complex)
    for row in range(count):
        for column in range(count):
            first, second = columns[row], columns[column]
            tensor[row, column] = np.vdot(first, second) - np.vdot(
                first, psi
            ) * np.vdot(psi, second)
    return tensor


_SUPPORTED_GATE_CASES = [
    ("rxx", 2, lambda circuit, angle: circuit.rxx(0, 1, angle)),
    ("ryy", 2, lambda circuit, angle: circuit.ryy(0, 1, angle)),
    ("rzz", 2, lambda circuit, angle: circuit.rzz(0, 1, angle)),
    ("rzx", 2, lambda circuit, angle: circuit.rzx(0, 1, angle)),
    ("crx", 2, lambda circuit, angle: circuit.crx(0, 1, angle)),
    ("cry", 2, lambda circuit, angle: circuit.cry(0, 1, angle)),
    ("crz", 2, lambda circuit, angle: circuit.crz(0, 1, angle)),
    (
        "c2rx",
        3,
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RX(angle).control(2), [0, 1, 2]
        ),
    ),
    (
        "c2ry",
        3,
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RY(angle).control(2), [0, 1, 2]
        ),
    ),
    (
        "c2rz",
        3,
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RZ(angle).control(2), [0, 1, 2]
        ),
    ),
    (
        "c2phase",
        3,
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.Phase(angle).control(2), [0, 1, 2]
        ),
    ),
]


def _gate_family_circuit(num_qubits, append_gate, theta):
    circuit = Circuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.ry(qubit, 0.21 + 0.17 * qubit)
        circuit.rz(qubit, -0.13 + 0.11 * qubit)
    append_gate(circuit, theta)
    for qubit in range(num_qubits):
        circuit.rx(qubit, 0.09 + 0.07 * qubit)
    return circuit


def test_single_rotation_metric_is_one_quarter(qgt) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",))
    result = qgt.compute(request)
    assert result.parameter_names == ("theta",)
    assert np.allclose(result.tensor, [[0.25 + 0j]], atol=1e-12)
    assert result.metric[0, 0] == pytest.approx(0.25, abs=1e-12)


def test_phase_gate_metric_is_one_quarter(qgt) -> None:
    phi = Parameter("phi")
    circuit = Circuit(1)
    circuit.h(0)
    circuit.phase(0, phi)
    request = _request(circuit, PauliString.from_str("X"), {"phi": 0.7}, ("phi",))
    result = qgt.compute(request)
    assert result.metric[0, 0] == pytest.approx(0.25, abs=1e-12)


def test_controlled_rotation_metric_is_one_quarter(qgt) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.x(0)
    circuit.crx(0, 1, theta)
    request = _request(circuit, PauliString.from_str("ZI"), {"theta": 0.5}, ("theta",))
    result = qgt.compute(request)
    assert result.metric[0, 0] == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize(
    ("gate_name", "num_qubits", "append_gate"),
    _SUPPORTED_GATE_CASES,
    ids=[case[0] for case in _SUPPORTED_GATE_CASES],
)
def test_supported_gate_families_match_independent_finite_difference(
    qgt,
    gate_name,
    num_qubits,
    append_gate,
) -> None:
    del gate_name
    theta = Parameter("theta")
    circuit = _gate_family_circuit(num_qubits, append_gate, theta)
    observable = PauliString.from_str("ZI" if num_qubits == 2 else "ZII")
    bindings = {"theta": 0.43}
    names = ("theta",)
    request = _request(circuit, observable, bindings, names)

    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)

    assert reference[0, 0].real > 1e-5
    assert np.allclose(result.tensor, reference, atol=1e-7, rtol=0.0)


def test_same_axis_parameters_produce_singular_metric(qgt) -> None:
    first = Parameter("first")
    second = Parameter("second")
    circuit = Circuit(1)
    circuit.ry(0, first)
    circuit.ry(0, second)
    bindings = {"first": 0.3, "second": -0.7}
    request = _request(
        circuit, PauliString.from_str("Z"), bindings, ("first", "second")
    )
    result = qgt.compute(request)
    expected = np.full((2, 2), 0.25)
    assert result.metric == pytest.approx(expected, abs=1e-12)
    assert abs(np.linalg.det(result.metric)) < 1e-12


def test_two_parameter_tensor_matches_finite_difference(qgt) -> None:
    first = Parameter("first")
    second = Parameter("second")
    circuit = Circuit(2)
    circuit.ry(0, first)
    circuit.cx(0, 1)
    circuit.ry(1, second)
    bindings = {"first": 0.3, "second": -0.7}
    names = ("first", "second")
    request = _request(circuit, PauliString.from_str("ZI"), bindings, names)
    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)
    assert np.allclose(result.tensor, reference, atol=1e-7)
    assert np.allclose(result.metric, reference.real, atol=1e-7)


def test_entangling_circuit_tensor_matches_finite_difference(qgt) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(1, theta)
    circuit.cx(0, 1)
    circuit.rzz(0, 1, theta)
    bindings = {"theta": 0.83}
    names = ("theta",)
    request = _request(circuit, PauliString.from_str("ZX"), bindings, names)
    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)
    assert np.allclose(result.tensor, reference, atol=1e-7)


def test_u_gate_all_slots_match_finite_difference(qgt) -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")
    circuit = Circuit(1)
    circuit.u(0, theta, phi, lam)
    bindings = {"theta": 0.2, "phi": 0.4, "lam": -0.1}
    names = ("theta", "phi", "lam")
    request = _request(circuit, PauliString.from_str("Y"), bindings, names)
    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)
    assert np.allclose(result.tensor, reference, atol=1e-7)


def test_u_slots_nested_in_larger_circuit_match_finite_difference(qgt) -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.u(1, theta, phi, 0.3)
    circuit.cx(0, 1)
    bindings = {"theta": 0.25, "phi": -0.6}
    names = ("theta", "phi")
    request = _request(circuit, PauliString.from_str("XZ"), bindings, names)
    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)
    assert np.allclose(result.tensor, reference, atol=1e-7)


def test_unrequested_unsupported_gate_parameter_allows_subset_qgt(qgt) -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.append_gate(StandardGate.RXY(phi, 0.23), [0])
    bindings = {"theta": 0.31, "phi": -0.47}
    names = ("theta",)
    request = _request(circuit, PauliString.from_str("Z"), bindings, names)

    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)

    assert np.allclose(result.tensor, reference, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("qubit_ids", [(7, 2), (1, 0)])
def test_u_gate_respects_noncontiguous_and_reordered_qubit_ids(qgt, qubit_ids) -> None:
    theta = Parameter("theta")
    first, second = (Qubit(index) for index in qubit_ids)
    circuit = Circuit([first, second])
    circuit.h(second)
    circuit.u(first, theta, 0.31, -0.27)
    bindings = {"theta": 0.42}
    names = ("theta",)
    request = _request(circuit, PauliString.from_str("ZI"), bindings, names)

    result = qgt.compute(request)
    reference = _difference_tensor(circuit, bindings, names)

    assert np.allclose(result.tensor, reference, atol=1e-7, rtol=0.0)


def test_global_phase_only_parameter_has_zero_row_and_column(qgt) -> None:
    theta = Parameter("theta")
    phase = Parameter("phase")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.set_global_phase(phase)
    bindings = {"theta": 0.4, "phase": 1.1}
    request = _request(circuit, PauliString.from_str("Z"), bindings, ("theta", "phase"))
    result = qgt.compute(request)
    assert result.metric[0, 0] == pytest.approx(0.25, abs=1e-12)
    assert result.metric[1, 1] == pytest.approx(0.0, abs=1e-12)
    assert result.metric[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_gphase_only_parameter_has_zero_row_and_column(qgt) -> None:
    theta = Parameter("theta")
    gphase = Parameter("gphase")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.append_gate(StandardGate.GPhase(gphase), [])
    bindings = {"theta": 0.4, "gphase": 0.9}
    request = _request(
        circuit, PauliString.from_str("Z"), bindings, ("theta", "gphase")
    )
    result = qgt.compute(request)
    assert result.metric[0, 0] == pytest.approx(0.25, abs=1e-12)
    assert result.metric[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_barrier_is_transparent_for_the_tensor(qgt) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.barrier([0, 1])
    circuit.ry(1, theta)
    circuit.barrier([0, 1])
    circuit.cx(0, 1)
    request = _request(circuit, PauliString.from_str("ZZ"), {"theta": 0.37}, ("theta",))
    result = qgt.compute(request)

    plain = Circuit(2)
    plain.h(0)
    plain.ry(1, theta)
    plain.cx(0, 1)
    plain_request = _request(
        plain, PauliString.from_str("ZZ"), {"theta": 0.37}, ("theta",)
    )
    plain_result = qgt.compute(plain_request)
    assert np.allclose(result.tensor, plain_result.tensor, atol=1e-12)


def test_empty_parameter_names_yield_empty_tensor(qgt) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())
    result = qgt.compute(request)
    assert result.parameter_names == ()
    assert result.tensor.shape == (0, 0)
    assert result.metric.shape == (0, 0)


@pytest.mark.parametrize("operation", ["measure", "reset", "delay"])
def test_empty_parameter_names_still_reject_nonunitary_circuits(qgt, operation) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    if operation == "measure":
        circuit.measure(0)
    elif operation == "reset":
        circuit.reset(0)
    else:
        circuit.delay(0, 1.0)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())

    with pytest.raises(ValueError, match="unitary"):
        qgt.compute(request)


def test_empty_parameter_names_still_reject_circuit_gates(qgt) -> None:
    inner = Circuit(1)
    inner.x(0)
    gate = CircuitGate("sub", FrozenCircuit([Qubit(0)], inner.operations), [])
    circuit = Circuit(1)
    circuit.append_circuit_gate(gate, [0], [])
    request = _request(circuit, PauliString.from_str("Z"), {}, ())

    with pytest.raises(ValueError, match="circuit gates"):
        qgt.compute(request)


def test_empty_parameter_names_allow_bound_symbolic_unitaries(qgt) -> None:
    theta = Parameter("theta")
    zero = SymbolicComplex.zero()
    one = SymbolicComplex.one()
    phase = SymbolicComplex.exp_i(theta)
    matrix = SymbolicMatrix([[one, zero], [zero, phase]])
    gate = UnitaryGate("phase_like", 1, 1).with_symbolic_matrix(matrix, ["theta"])
    circuit = Circuit(1)
    circuit.append_unitary_gate(gate, [0], [theta])
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())

    result = qgt.compute(request)

    assert result.parameter_names == ()
    assert result.tensor.shape == (0, 0)
    assert result.metric.shape == (0, 0)


def test_compute_rejects_non_gradient_request(qgt) -> None:
    with pytest.raises(TypeError, match="GradientRequest"):
        qgt.compute(object())  # type: ignore[arg-type]


def test_compute_rejects_measurement_circuit(qgt) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.measure(0)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ("theta",))
    with pytest.raises(ValueError, match="unitary"):
        qgt.compute(request)


def test_result_rejects_non_hermitian_tensor() -> None:
    tensor = np.array([[0.0 + 0j, 1.0 + 0j], [0.0 + 0j, 0.0 + 0j]])
    with pytest.raises(ValueError, match="Hermitian"):
        QGTResult(("a", "b"), tensor, tensor.real)


def test_result_uses_absolute_tolerance_for_large_non_hermitian_tensor() -> None:
    tensor = np.array([[1.0e8 + 0j, 500.0 + 0j], [0.0 + 0j, 1.0e8 + 0j]])
    with pytest.raises(ValueError, match="Hermitian"):
        QGTResult(("a", "b"), tensor, tensor.real)


def test_result_rejects_metric_mismatching_tensor_real_part() -> None:
    tensor = np.array([[0.25 + 0j]])
    with pytest.raises(ValueError, match="real part"):
        QGTResult(("a",), tensor, np.array([[0.5]]))


def test_result_rejects_complex_metric_without_silent_conversion() -> None:
    tensor = np.array([[0.25 + 0.0j]])
    metric = np.array([[0.25 + 100.0j]])

    with pytest.raises(ValueError, match="real-valued"):
        QGTResult(("a",), tensor, metric)


def test_result_uses_absolute_tolerance_for_large_metric_mismatch() -> None:
    tensor = np.array([[1.0e8 + 0j]])
    with pytest.raises(ValueError, match="real part"):
        QGTResult(("a",), tensor, np.array([[1.0e8 + 500.0]]))


def test_result_rejects_duplicate_names_and_bad_shapes() -> None:
    tensor = np.eye(2)
    with pytest.raises(ValueError, match="unique"):
        QGTResult(("a", "a"), tensor, tensor.real)
    with pytest.raises(ValueError, match="square"):
        QGTResult(("a",), np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="square"):
        QGTResult(("a",), np.eye(1), np.zeros((1, 2)))


def test_result_copies_input_arrays() -> None:
    tensor = np.array([[0.25 + 0j]])
    metric = np.array([[0.25]])
    result = QGTResult(("a",), tensor, metric)
    tensor[0, 0] = 99.0
    assert result.tensor[0, 0] == pytest.approx(0.25, abs=1e-15)
