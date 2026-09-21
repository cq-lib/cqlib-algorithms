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

"""Parameter-shift gradient tests against analytic and difference references."""

from __future__ import annotations

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit.gates import StandardGate
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


@pytest.fixture
def gradient() -> ParameterShiftGradient:
    return ParameterShiftGradient()


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


def _statevector_in_insertion_order(circuit) -> np.ndarray:
    qubit_order = [qubit.index for qubit in circuit.qubits]
    return circuit.to_matrix(qubit_order)[:, 0]


_SUPPORTED_GATE_CASES = [
    ("rxx", 2, "XY", lambda circuit, angle: circuit.rxx(0, 1, angle)),
    ("ryy", 2, "YX", lambda circuit, angle: circuit.ryy(0, 1, angle)),
    ("rzz", 2, "YZ", lambda circuit, angle: circuit.rzz(0, 1, angle)),
    ("rzx", 2, "YZ", lambda circuit, angle: circuit.rzx(0, 1, angle)),
    ("crx", 2, "XY", lambda circuit, angle: circuit.crx(0, 1, angle)),
    ("cry", 2, "YY", lambda circuit, angle: circuit.cry(0, 1, angle)),
    ("crz", 2, "ZY", lambda circuit, angle: circuit.crz(0, 1, angle)),
    (
        "c2rx",
        3,
        "XXY",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RX(angle).control(2), [0, 1, 2]
        ),
    ),
    (
        "c2ry",
        3,
        "YXY",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RY(angle).control(2), [0, 1, 2]
        ),
    ),
    (
        "c2rz",
        3,
        "ZXY",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RZ(angle).control(2), [0, 1, 2]
        ),
    ),
    (
        "c2phase",
        3,
        "XXY",
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


def test_single_ry_gradient_matches_negative_sine(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.parameter_names == ("theta",)
    assert result.values == pytest.approx((-math.sin(0.4),), abs=1e-12)


def test_single_rx_gradient_matches_cosine_derivative(
    gradient, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.rx(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": -0.2}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.values == pytest.approx((-math.sin(-0.2),), abs=1e-12)


def test_phase_gate_gradient_matches_cosine(gradient, matrix_estimator) -> None:
    phi = Parameter("phi")
    circuit = Circuit(1)
    circuit.h(0)
    circuit.phase(0, phi)
    request = _request(circuit, PauliString.from_str("X"), {"phi": 0.7}, ("phi",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.values == pytest.approx((-math.sin(0.7),), abs=1e-12)


def test_controlled_rx_gradient_with_control_in_one_state(
    gradient, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.x(0)
    circuit.crx(0, 1, theta)
    request = _request(circuit, PauliString.from_str("ZI"), {"theta": 0.5}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.values == pytest.approx((-math.sin(0.5),), abs=1e-12)


def test_controlled_rx_gradient_with_control_in_zero_state_is_zero(
    gradient, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.crx(0, 1, theta)
    request = _request(circuit, PauliString.from_str("ZI"), {"theta": 0.5}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.values == pytest.approx((0.0,), abs=1e-12)


def test_mcgate_ry_gradient_matches_analytic(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.x(0)
    circuit.append_mc_gate(StandardGate.RY(theta).control(1), [0, 1])
    request = _request(circuit, PauliString.from_str("ZI"), {"theta": 0.5}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.values == pytest.approx((-math.sin(0.5),), abs=1e-12)


@pytest.mark.parametrize(
    ("gate_name", "num_qubits", "observable_label", "append_gate"),
    _SUPPORTED_GATE_CASES,
    ids=[case[0] for case in _SUPPORTED_GATE_CASES],
)
def test_supported_gate_families_match_independent_finite_difference(
    gradient,
    matrix_estimator,
    finite_difference_step,
    gate_name,
    num_qubits,
    observable_label,
    append_gate,
) -> None:
    del gate_name
    theta = Parameter("theta")
    circuit = _gate_family_circuit(num_qubits, append_gate, theta)
    observable = PauliString.from_str(observable_label)
    value = 0.43
    request = _request(circuit, observable, {"theta": value}, ("theta",))

    result = gradient.run(matrix_estimator, [request])[0]

    def expectation(shift: float) -> float:
        bound = circuit.assign_parameters({"theta": value + shift})
        state = _statevector_in_insertion_order(bound)
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    step = finite_difference_step
    expected = (expectation(step) - expectation(-step)) / (2.0 * step)
    assert abs(expected) > 1e-5
    assert result.values == pytest.approx((expected,), abs=1e-7)


def test_u_theta_slot_gradient_matches_ry_reference(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.u(0, theta, 0.0, 0.0)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.values == pytest.approx((-math.sin(0.3),), abs=1e-12)


def test_affine_expression_applies_chain_rule(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, 2 * theta + 1)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.25}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    expected = -2.0 * math.sin(2 * 0.25 + 1)
    assert result.values == pytest.approx((expected,), abs=1e-12)


def test_shared_symbol_two_rotations_matches_finite_difference(
    gradient, matrix_estimator, finite_difference_step
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(1, theta)
    circuit.cx(0, 1)
    circuit.ry(0, theta)
    observable = PauliString.from_str("ZZ")
    value = 0.31
    request = _request(circuit, observable, {"theta": value}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]

    def expectation(shift: float) -> float:
        bound = circuit.assign_parameters({"theta": value + shift})
        state = _statevector_in_insertion_order(bound)
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    step = finite_difference_step
    expected = (expectation(step) - expectation(-step)) / (2 * step)
    assert result.values == pytest.approx((expected,), abs=1e-7)


def test_shared_symbol_rotation_and_phase_matches_finite_difference(
    gradient, matrix_estimator, finite_difference_step
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(1, theta)
    circuit.cx(0, 1)
    circuit.phase(1, theta)
    observable = PauliString.from_str("ZZ")
    value = 0.31
    request = _request(circuit, observable, {"theta": value}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]

    def expectation(shift: float) -> float:
        bound = circuit.assign_parameters({"theta": value + shift})
        state = _statevector_in_insertion_order(bound)
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    step = finite_difference_step
    expected = (expectation(step) - expectation(-step)) / (2 * step)
    assert result.values == pytest.approx((expected,), abs=1e-7)


def test_shared_symbol_three_occurrences_matches_finite_difference(
    gradient, matrix_estimator, finite_difference_step
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.ry(0, theta)
    circuit.ry(1, theta)
    circuit.rzz(0, 1, theta)
    observable = PauliString.from_str("ZX")
    value = -0.44
    request = _request(circuit, observable, {"theta": value}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]

    def expectation(shift: float) -> float:
        bound = circuit.assign_parameters({"theta": value + shift})
        state = _statevector_in_insertion_order(bound)
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    step = finite_difference_step
    expected = (expectation(step) - expectation(-step)) / (2 * step)
    assert result.values == pytest.approx((expected,), abs=1e-7)


def test_hamiltonian_observable_gradient_matches_analytic_sum(
    gradient, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    hamiltonian = Hamiltonian.from_list(
        [(PauliString.from_str("Z"), 1.0), (PauliString.from_str("X"), 0.5)]
    )
    request = _request(circuit, hamiltonian, {"theta": 0.6}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]
    expected = -math.sin(0.6) + 0.5 * math.cos(0.6)
    assert result.values == pytest.approx((expected,), abs=1e-12)


@pytest.mark.parametrize("scale", [1.0, 1.0e6])
def test_hamiltonian_coefficient_scale_preserves_gradient_linearity(
    gradient, matrix_estimator, scale
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    hamiltonian = Hamiltonian.from_list([(PauliString.from_str("Z"), scale)])
    value = 0.37
    request = _request(circuit, hamiltonian, {"theta": value}, ("theta",))

    result = gradient.run(matrix_estimator, [request])[0]

    expected = -scale * math.sin(value)
    assert result.values == pytest.approx((expected,), rel=1e-11, abs=1e-10)


def test_global_phase_only_parameter_has_zero_gradient_without_evaluations(
    constant_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.x(0)
    circuit.set_global_phase(theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.9}, ("theta",))
    estimator = constant_estimator_factory(3.0)
    result = ParameterShiftGradient().run(estimator, [request])[0]
    assert result.values == (0.0,)
    assert estimator.calls == 0


def test_all_evaluations_flatten_into_a_single_estimator_batch(
    recording_estimator_factory,
) -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    first = Circuit(1)
    first.ry(0, theta)
    second = Circuit(2)
    second.h(0)
    second.ry(1, phi)
    second.cx(0, 1)
    second.phase(1, phi)
    requests = [
        _request(first, PauliString.from_str("Z"), {"theta": 0.1}, ("theta",)),
        _request(second, PauliString.from_str("ZZ"), {"phi": 0.2}, ("phi",)),
    ]
    recording = recording_estimator_factory()
    ParameterShiftGradient().run(recording, requests)
    assert recording.batch_sizes == [2 + 5]


def test_multiple_parameters_in_one_request_are_ordered(
    gradient, matrix_estimator
) -> None:
    first = Parameter("first")
    second = Parameter("second")
    circuit = Circuit(2)
    circuit.ry(0, first)
    circuit.ry(1, second)
    # "ZI" measures Z on the higher-index qubit, driven by ``second``.
    request = _request(
        circuit,
        PauliString.from_str("ZI"),
        {"first": 0.3, "second": -0.8},
        ("first", "second"),
    )
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.parameter_names == ("first", "second")
    assert result.values == pytest.approx((0.0, -math.sin(-0.8)), abs=1e-12)


def test_empty_parameter_names_return_empty_gradient(
    gradient, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())
    result = gradient.run(matrix_estimator, [request])[0]
    assert result.parameter_names == ()
    assert result.values == ()


def test_empty_parameter_names_still_reject_nonunitary_circuit(
    gradient, matrix_estimator
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.reset(0)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())

    with pytest.raises(ValueError, match="unitary"):
        gradient.run(matrix_estimator, [request])


def test_empty_request_sequence_returns_empty_list(gradient, matrix_estimator) -> None:
    assert gradient.run(matrix_estimator, []) == []


def test_results_preserve_request_order(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    first = Circuit(1)
    first.ry(0, theta)
    second = Circuit(1)
    second.rx(0, theta)
    requests = [
        _request(first, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",)),
        _request(second, PauliString.from_str("Z"), {"theta": 0.5}, ("theta",)),
    ]
    results = gradient.run(matrix_estimator, requests)
    assert len(results) == 2
    assert results[0].values == pytest.approx((-math.sin(0.2),), abs=1e-12)
    assert results[1].values == pytest.approx((-math.sin(0.5),), abs=1e-12)


def test_rejects_non_estimator_object() -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",))
    with pytest.raises(TypeError, match="estimator"):
        ParameterShiftGradient().run(object(), [request])  # type: ignore[arg-type]


def test_rejects_non_gradient_request_items(gradient, matrix_estimator) -> None:
    with pytest.raises(TypeError, match="GradientRequest"):
        gradient.run(matrix_estimator, [object()])  # type: ignore[list-item]


def test_rejects_unsupported_gate_before_estimator_calls(
    constant_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.append_gate(StandardGate.XY2M(theta), [0])
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",))
    estimator = constant_estimator_factory(1.0)
    with pytest.raises(ValueError, match="not supported"):
        ParameterShiftGradient().run(estimator, [request])
    assert estimator.calls == 0


def test_unrequested_unsupported_gate_parameter_allows_subset_gradient(
    gradient, matrix_estimator, finite_difference_step
) -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    circuit.append_gate(StandardGate.RXY(phi, 0.23), [0])
    observable = PauliString.from_str("Z")
    bindings = {"theta": 0.31, "phi": -0.47}
    request = _request(circuit, observable, bindings, ("theta",))

    result = gradient.run(matrix_estimator, [request])[0]

    def expectation(shift: float) -> float:
        shifted = dict(bindings)
        shifted["theta"] += shift
        bound = circuit.assign_parameters(shifted)
        state = _statevector_in_insertion_order(bound)
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    step = finite_difference_step
    expected = (expectation(step) - expectation(-step)) / (2.0 * step)
    assert result.values == pytest.approx((expected,), abs=1e-7)


def test_rejects_result_count_mismatch(matrix_estimator_factory) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",))

    class ShortEstimator(matrix_estimator_factory):
        __slots__ = ()

        def run(self, requests, /):
            return super().run(list(requests)[:-1])

    with pytest.raises(ValueError, match="one result per evaluation"):
        ParameterShiftGradient().run(ShortEstimator(), [request])


def test_imaginary_residual_guard_accepts_real_evaluations(matrix_estimator) -> None:
    """All reachable shift plans carry real weights, so an honest
    estimator assembles a real gradient; the imaginary-residual guard
    is defense in depth and must not fire for legitimate evaluations."""
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.ry(1, theta)
    circuit.cx(0, 1)
    circuit.phase(1, theta)
    request = _request(circuit, PauliString.from_str("ZZ"), {"theta": 0.31}, ("theta",))
    result = ParameterShiftGradient().run(matrix_estimator, [request])[0]
    assert isinstance(result.values[0], float)


def test_barrier_does_not_change_the_gradient(gradient, matrix_estimator) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.barrier([0, 1])
    circuit.ry(1, theta)
    circuit.barrier([0, 1])
    circuit.cx(0, 1)
    request = _request(circuit, PauliString.from_str("ZZ"), {"theta": 0.37}, ("theta",))
    result = gradient.run(matrix_estimator, [request])[0]

    plain = Circuit(2)
    plain.h(0)
    plain.ry(1, theta)
    plain.cx(0, 1)
    plain_request = _request(
        plain, PauliString.from_str("ZZ"), {"theta": 0.37}, ("theta",)
    )
    plain_result = gradient.run(matrix_estimator, [plain_request])[0]
    assert result.values == pytest.approx(plain_result.values, abs=1e-12)


def test_shifted_requests_reuse_the_original_circuit_object(
    matrix_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.ry(0, theta)
    request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",))

    seen_circuits: list[int] = []

    class InspectingEstimator(matrix_estimator_factory):
        __slots__ = ()

        def run(self, requests, /):
            seen_circuits.extend(id(item.circuit) for item in requests)
            return super().run(requests)

    ParameterShiftGradient().run(InspectingEstimator(), [request])
    assert seen_circuits
    assert {identity for identity in seen_circuits} == {id(circuit)}
