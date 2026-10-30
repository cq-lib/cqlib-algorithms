# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# use this file except in compliance with the License, obtained from the
# LICENSE.txt file in the root directory of this source tree or
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Finite-difference gradient tests against analytic and parameter-shift references."""

from __future__ import annotations

import math

import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit.gates import StandardGate
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import (
    FiniteDifferenceGradient,
    ParameterShiftGradient,
)
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


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


class TestConstruction:
    def test_default_construction(self) -> None:
        gradient = FiniteDifferenceGradient()
        assert gradient.epsilon == 1e-6
        assert gradient.method == "central"

    def test_custom_construction(self) -> None:
        gradient = FiniteDifferenceGradient(epsilon=1e-4, method="forward")
        assert gradient.epsilon == 1e-4
        assert gradient.method == "forward"

    def test_epsilon_must_be_positive(self) -> None:
        for epsilon in (0.0, -1e-6, math.inf, math.nan):
            with pytest.raises(ValueError, match="finite positive"):
                FiniteDifferenceGradient(epsilon=epsilon)

    def test_epsilon_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="real number"):
            FiniteDifferenceGradient(epsilon="small")  # type: ignore[arg-type]

    def test_method_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="forward.*central"):
            FiniteDifferenceGradient(method="backward")

    def test_method_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="string"):
            FiniteDifferenceGradient(method=1)  # type: ignore[arg-type]


class TestAnalyticValues:
    def test_single_ry_central_matches_negative_sine(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",)
        )
        gradient = FiniteDifferenceGradient(epsilon=1e-6)
        result = gradient.run(matrix_estimator, [request])[0]
        assert result.parameter_names == ("theta",)
        assert result.values == pytest.approx((-math.sin(0.4),), abs=1e-9)

    def test_single_ry_forward_matches_negative_sine(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",)
        )
        gradient = FiniteDifferenceGradient(epsilon=1e-6, method="forward")
        result = gradient.run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx((-math.sin(0.4),), abs=1e-5)

    def test_forward_error_is_first_order(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",)
        )
        exact = -math.sin(0.4)
        errors = []
        for epsilon in (1e-3, 1e-2):
            gradient = FiniteDifferenceGradient(epsilon=epsilon, method="forward")
            result = gradient.run(matrix_estimator, [request])[0]
            errors.append(abs(result.values[0] - exact))
        # The first-order error must shrink as the step shrinks.
        assert errors[0] < errors[1]
        assert errors[0] / errors[1] < 0.5

    def test_central_beats_forward_at_same_step(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",)
        )
        exact = -math.sin(0.4)
        central = FiniteDifferenceGradient(epsilon=1e-4)
        forward = FiniteDifferenceGradient(epsilon=1e-4, method="forward")
        central_error = abs(
            central.run(matrix_estimator, [request])[0].values[0] - exact
        )
        forward_error = abs(
            forward.run(matrix_estimator, [request])[0].values[0] - exact
        )
        assert central_error < forward_error

    def test_two_parameter_gradient(self, matrix_estimator) -> None:
        first = Parameter("a")
        second = Parameter("b")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.ry(1, second)
        circuit.cx(0, 1)
        request = _request(
            circuit,
            PauliString.from_str("ZZ"),
            {"a": 0.3, "b": -0.7},
            ("a", "b"),
        )
        gradient = FiniteDifferenceGradient()
        result = gradient.run(matrix_estimator, [request])[0]
        reference = ParameterShiftGradient().run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx(reference.values, abs=1e-9)


class TestUnsupportedGates:
    def test_fd_works_where_parameter_shift_fails(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(2)
        circuit.ry(0, 0.2)
        circuit.ry(1, 0.3)
        circuit.xy(1, theta)
        request = _request(
            circuit, PauliString.from_str("ZY"), {"theta": 0.5}, ("theta",)
        )
        gradient = FiniteDifferenceGradient(epsilon=1e-5)
        result = gradient.run(matrix_estimator, [request])[0]
        coarse = FiniteDifferenceGradient(epsilon=1e-4)
        reference = coarse.run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx(reference.values, abs=1e-3)
        with pytest.raises(ValueError):
            ParameterShiftGradient().run(matrix_estimator, [request])


@pytest.mark.parametrize(
    ("gate_name", "num_qubits", "pauli", "append_gate"),
    _SUPPORTED_GATE_CASES,
)
class TestParameterShiftAgreement:
    def test_matches_parameter_shift(
        self, matrix_estimator, gate_name, num_qubits, pauli, append_gate
    ) -> None:
        theta = Parameter("theta")
        circuit = _gate_family_circuit(num_qubits, append_gate, theta)
        request = _request(
            circuit,
            PauliString.from_str(pauli),
            {"theta": 0.66},
            ("theta",),
        )
        fd = FiniteDifferenceGradient(epsilon=1e-6).run(matrix_estimator, [request])[0]
        ps = ParameterShiftGradient().run(matrix_estimator, [request])[0]
        assert fd.values == pytest.approx(ps.values, abs=1e-7)


class TestBatchingAndValidation:
    def test_single_flattened_batch(self, recording_estimator_factory) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        first = _request(
            Circuit(1).ry(0, theta) or _single_ry(theta),
            PauliString.from_str("Z"),
            {"theta": 0.2},
            ("theta",),
        )
        second = _request(
            _single_ry(phi),
            PauliString.from_str("Z"),
            {"phi": -0.4},
            ("phi",),
        )
        estimator = recording_estimator_factory()
        gradient = FiniteDifferenceGradient()
        results = gradient.run(estimator, [first, second])
        assert estimator.batch_sizes == [4]  # 2 requests x 2 evaluations
        assert len(results) == 2

    def test_empty_parameter_names_skip_estimator(
        self, constant_estimator_factory
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ())
        estimator = constant_estimator_factory(1.0)
        gradient = FiniteDifferenceGradient()
        result = gradient.run(estimator, [request])[0]
        assert result.parameter_names == ()
        assert result.values == ()
        assert estimator.calls == 0

    def test_forward_difference_uses_two_evaluations_per_name(
        self, recording_estimator_factory
    ) -> None:
        theta = Parameter("theta")
        request = _request(
            _single_ry(theta),
            PauliString.from_str("Z"),
            {"theta": 0.2},
            ("theta",),
        )
        estimator = recording_estimator_factory()
        gradient = FiniteDifferenceGradient(method="forward")
        gradient.run(estimator, [request])
        assert estimator.batch_sizes == [2]

    def test_estimator_result_count_mismatch(
        self, constant_estimator_factory, exact_result_factory
    ) -> None:
        theta = Parameter("theta")
        request = _request(
            _single_ry(theta),
            PauliString.from_str("Z"),
            {"theta": 0.2},
            ("theta",),
        )

        class Broken(constant_estimator_factory):
            def run(self, requests, /):
                self.calls += 1
                return [
                    exact_result_factory(1.0) for _ in range(len(list(requests)) - 1)
                ]

        with pytest.raises(ValueError, match="result count|per evaluation"):
            FiniteDifferenceGradient().run(Broken(1.0), [request])

    def test_estimator_type_is_checked(self, matrix_estimator) -> None:
        request = _request(
            _single_ry(Parameter("theta")),
            PauliString.from_str("Z"),
            {"theta": 0.2},
            ("theta",),
        )
        with pytest.raises(TypeError, match="Estimator"):
            FiniteDifferenceGradient().run(object(), [request])  # type: ignore[arg-type]

    def test_request_type_is_checked(self, matrix_estimator) -> None:
        with pytest.raises(TypeError, match="GradientRequest"):
            FiniteDifferenceGradient().run(matrix_estimator, [object()])  # type: ignore[list-item]


def _single_ry(symbol) -> Circuit:
    circuit = Circuit(1)
    circuit.ry(0, symbol)
    return circuit
