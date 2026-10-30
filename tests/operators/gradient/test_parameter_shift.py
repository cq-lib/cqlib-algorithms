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
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import ClassicalType
from cqlib.circuit.gates import StandardGate
from cqlib.qis import Hamiltonian, Pauli, PauliString

from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.optimizer import Adam
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    GradientRequest,
    OptimizerRequest,
)


@pytest.fixture
def gradient() -> ParameterShiftGradient:
    return ParameterShiftGradient()


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


def _statevector_in_insertion_order(circuit) -> np.ndarray:
    qubit_order = [qubit.index for qubit in circuit.qubits]
    return circuit.to_matrix(qubit_order)[:, 0]


_WEIGHTED_EDGES = (
    (0, 1, 0.37),
    (0, 2, 0.61),
    (0, 5, 0.83),
    (1, 2, 1.07),
    (1, 3, 0.43),
    (1, 5, 0.79),
    (2, 3, 0.97),
    (2, 4, 1.13),
    (3, 4, 0.53),
    (3, 5, 1.21),
    (4, 5, 0.71),
)


def _weighted_maxcut_hamiltonian(*, sign: float = 1.0) -> Hamiltonian:
    observable = Hamiltonian(6)
    identity = PauliString(6)
    for first, second, weight in _WEIGHTED_EDGES:
        zz = PauliString(6)
        zz.set_pauli(first, Pauli.z())
        zz.set_pauli(second, Pauli.z())
        observable.add_term(identity, sign * weight / 2.0)
        observable.add_term(zz, -sign * weight / 2.0)
    return observable


def _qaoa_style_circuit(reps: int = 2) -> Circuit:
    circuit = Circuit(6)
    for qubit in range(6):
        circuit.h(qubit)
    for layer in range(reps):
        gamma = Parameter(f"gamma_{layer}")
        beta = Parameter(f"beta_{layer}")
        for first, second, weight in _WEIGHTED_EDGES:
            circuit.rzz(first, second, -weight * gamma)
        for qubit in range(6):
            circuit.rx(qubit, 2.0 * beta)
    return circuit


def _independent_central_difference(
    circuit: Circuit,
    observable: Hamiltonian | PauliString,
    bindings: dict[str, float],
    name: str,
    step: float = 1e-6,
) -> float:
    def expectation(offset: float) -> float:
        shifted = dict(bindings)
        shifted[name] += offset
        state = _statevector_in_insertion_order(circuit.assign_parameters(shifted))
        return float(np.real(np.vdot(state, observable.to_matrix() @ state)))

    return (expectation(step) - expectation(-step)) / (2.0 * step)


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


def test_eleven_decimal_weight_occurrences_use_twenty_two_evaluations(
    recording_estimator_factory,
) -> None:
    gamma = Parameter("gamma")
    circuit = Circuit(6)
    for qubit in range(6):
        circuit.h(qubit)
    for first, second, weight in _WEIGHTED_EDGES:
        circuit.rzz(first, second, -weight * gamma)
    for qubit in range(6):
        circuit.rx(qubit, 0.19 + 0.03 * qubit)
    observable = _weighted_maxcut_hamiltonian()
    bindings = {"gamma": 0.31}
    request = _request(circuit, observable, bindings, ("gamma",))
    estimator = recording_estimator_factory()

    result = ParameterShiftGradient().run(estimator, [request])[0]

    expected = _independent_central_difference(circuit, observable, bindings, "gamma")
    assert estimator.batch_sizes == [22]
    assert abs(expected) > 1e-4
    assert result.values == pytest.approx((expected,), abs=2e-7)


@pytest.mark.slow
def test_two_layer_weighted_qaoa_style_gradient_matches_all_finite_differences(
    recording_estimator_factory,
) -> None:
    circuit = _qaoa_style_circuit()
    observable = _weighted_maxcut_hamiltonian()
    names = ("gamma_0", "beta_0", "gamma_1", "beta_1")
    bindings = {
        "gamma_0": 0.31,
        "beta_0": 0.22,
        "gamma_1": -0.17,
        "beta_1": 0.41,
    }
    estimator = recording_estimator_factory()
    request = _request(circuit, observable, bindings, names)

    result = ParameterShiftGradient().run(estimator, [request])[0]

    expected = tuple(
        _independent_central_difference(circuit, observable, bindings, name)
        for name in names
    )
    assert estimator.batch_sizes == [68]
    assert max(abs(value) for value in expected) > 1e-3
    assert result.values == pytest.approx(expected, abs=3e-7)


@pytest.mark.slow
def test_adam_with_parameter_shift_improves_weighted_qaoa_style_objective(
    matrix_estimator,
) -> None:
    circuit = _qaoa_style_circuit()
    observable = _weighted_maxcut_hamiltonian(sign=-1.0)
    names = ("gamma_0", "beta_0", "gamma_1", "beta_1")
    bindings = {
        "gamma_0": 0.31,
        "beta_0": 0.22,
        "gamma_1": -0.17,
        "beta_1": 0.41,
    }
    request = OptimizerRequest(
        EstimatorRequest(circuit, observable, bindings),
        names,
    )
    initial = matrix_estimator.run([request.estimator_request])[0].expectation

    result = Adam(
        ParameterShiftGradient(),
        learning_rate=0.04,
        maxiter=4,
    ).run(matrix_estimator, [request])[0]

    assert result.iterations == 4
    assert result.optimal_value < initial - 1e-3


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
    assert recording.batch_sizes == [2 + 4]


def test_local_evaluations_keep_occurrence_then_positive_negative_order(
    matrix_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.ry(0, theta)
    circuit.rz(1, theta)
    request = _request(
        circuit,
        PauliString.from_str("ZZ"),
        {"theta": 0.2},
        ("theta",),
    )
    seen_angles: list[tuple[float, float]] = []

    class InspectingEstimator(matrix_estimator_factory):
        __slots__ = ()

        def run(self, requests, /):
            request_list = list(requests)
            for item in request_list:
                seen_angles.append(
                    (
                        float(item.circuit.operations[0].params[0]),
                        float(item.circuit.operations[1].params[0]),
                    )
                )
            return super().run(request_list)

    ParameterShiftGradient().run(InspectingEstimator(), [request])

    shift = math.pi / 2.0
    assert seen_angles == pytest.approx(
        [
            (0.2 + shift, 0.2),
            (0.2 - shift, 0.2),
            (0.2, 0.2 + shift),
            (0.2, 0.2 - shift),
        ],
        abs=1e-15,
    )


def test_controlled_rotation_uses_four_evaluations(
    recording_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.crx(0, 1, theta)
    request = _request(
        circuit,
        PauliString.from_str("ZI"),
        {"theta": 0.2},
        ("theta",),
    )
    estimator = recording_estimator_factory()

    ParameterShiftGradient().run(estimator, [request])

    assert estimator.batch_sizes == [4]


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


def test_shifted_requests_preserve_structure_without_mutating_original(
    matrix_estimator_factory,
) -> None:
    theta = Parameter("theta")
    first = Qubit(5)
    second = Qubit(9)
    circuit = Circuit([first, second])
    circuit.h(first)
    circuit.barrier([first, second])
    circuit.append_gate(StandardGate.RY(theta), [second], "shifted-slot")
    circuit.set_global_phase(0.37)
    original_operations = tuple(str(operation) for operation in circuit.operations)
    original_symbols = tuple(circuit.used_symbols)
    request = _request(
        circuit,
        PauliString.from_str("ZI"),
        {"theta": 0.2},
        ("theta",),
    )

    seen = []

    class InspectingEstimator(matrix_estimator_factory):
        __slots__ = ()

        def run(self, requests, /):
            seen.extend(requests)
            return super().run(requests)

    ParameterShiftGradient().run(InspectingEstimator(), [request])
    assert len(seen) == 2
    assert all(item.circuit is not circuit for item in seen)
    assert all(
        [qubit.index for qubit in item.circuit.qubits] == [5, 9] for item in seen
    )
    assert all(item.circuit.global_phase.evaluate() == 0.37 for item in seen)
    assert all(item.circuit.operations[1].name == "barrier" for item in seen)
    assert all(item.circuit.operations[2].label == "shifted-slot" for item in seen)
    assert all(item.parameter_bindings == {} for item in seen)
    assert (
        tuple(str(operation) for operation in circuit.operations) == original_operations
    )
    assert tuple(circuit.used_symbols) == original_symbols
    assert circuit.global_phase.evaluate() == 0.37


def test_shifted_requests_preserve_unused_classical_metadata(
    matrix_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(1)
    circuit.var(ClassicalType.bool())
    circuit.ry(0, theta)
    circuit.set_global_phase(0.23)
    source_vars = list(circuit.classical_vars)
    source_values = list(circuit.classical_values)
    source_operations = tuple(str(operation) for operation in circuit.operations)
    bound = circuit.assign_parameters({"theta": 0.4})
    request = _request(
        circuit,
        PauliString.from_str("Z"),
        {"theta": 0.4},
        ("theta",),
    )
    shifted_circuits: list[Circuit] = []

    class InspectingEstimator(matrix_estimator_factory):
        __slots__ = ()

        def run(self, requests, /):
            request_list = list(requests)
            shifted_circuits.extend(item.circuit for item in request_list)
            return super().run(request_list)

    ParameterShiftGradient().run(InspectingEstimator(), [request])

    assert source_vars == [ClassicalType.bool()]
    assert bound.classical_vars == source_vars
    assert bound.classical_values == source_values
    assert len(shifted_circuits) == 2
    assert all(item.classical_vars == source_vars for item in shifted_circuits)
    assert all(item.classical_values == source_values for item in shifted_circuits)
    assert all(item.global_phase.evaluate() == 0.23 for item in shifted_circuits)
    assert circuit.classical_vars == source_vars
    assert circuit.classical_values == source_values
    assert (
        tuple(str(operation) for operation in circuit.operations) == source_operations
    )


def test_shots_are_preserved_for_every_local_evaluation(
    constant_estimator_factory,
) -> None:
    theta = Parameter("theta")
    circuit = Circuit(2)
    circuit.ry(0, theta)
    circuit.ry(1, theta)
    request = GradientRequest(
        EstimatorRequest(
            circuit,
            PauliString.from_str("ZZ"),
            {"theta": 0.2},
            2048,
        ),
        ("theta",),
    )

    class InspectingEstimator(constant_estimator_factory):
        __slots__ = ("seen_shots",)

        def __init__(self) -> None:
            super().__init__(0.25)
            self.seen_shots: list[int | None] = []

        def run(self, requests, /):
            request_list = list(requests)
            self.seen_shots.extend(item.shots for item in request_list)
            return super().run(request_list)

    estimator = InspectingEstimator()

    result = ParameterShiftGradient().run(estimator, [request])[0]

    assert result.values == pytest.approx((0.0,), abs=1e-15)
    assert estimator.sizes == [4]
    assert estimator.seen_shots == [2048] * 4
