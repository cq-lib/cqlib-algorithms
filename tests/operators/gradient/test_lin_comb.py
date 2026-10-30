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

"""Linear-combination gradient tests against analytic and parameter-shift references."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit.gates import StandardGate
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms.operators.gradient import (
    LinCombEstimatorGradient,
    ParameterShiftGradient,
)
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


def _central_difference(estimator, circuit, observable, bindings, name, step=1e-6):
    plus = dict(bindings)
    minus = dict(bindings)
    plus[name] += step
    minus[name] -= step
    results = estimator.run(
        [
            EstimatorRequest(circuit, observable, plus),
            EstimatorRequest(circuit, observable, minus),
        ]
    )
    return (results[0].expectation - results[1].expectation) / (2.0 * step)


_SUPPORTED_GATE_CASES = [
    ("rx", 1, "Z", lambda circuit, angle: circuit.rx(0, angle)),
    ("ry", 1, "Z", lambda circuit, angle: circuit.ry(0, angle)),
    ("rz", 1, "X", lambda circuit, angle: circuit.rz(0, angle)),
    ("phase", 1, "X", lambda circuit, angle: circuit.phase(0, angle)),
    ("rxx", 2, "XY", lambda circuit, angle: circuit.rxx(0, 1, angle)),
    ("ryy", 2, "YX", lambda circuit, angle: circuit.ryy(0, 1, angle)),
    ("rzz", 2, "YZ", lambda circuit, angle: circuit.rzz(0, 1, angle)),
    ("rzx", 2, "YZ", lambda circuit, angle: circuit.rzx(0, 1, angle)),
    ("crx", 2, "XY", lambda circuit, angle: circuit.crx(0, 1, angle)),
    ("cry", 2, "YY", lambda circuit, angle: circuit.cry(0, 1, angle)),
    ("crz", 2, "ZY", lambda circuit, angle: circuit.crz(0, 1, angle)),
    (
        "cphase",
        2,
        "XY",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.Phase(angle).control(1), [0, 1]
        ),
    ),
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
    (
        "c3rx",
        4,
        "YXYX",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RX(angle).control(3), [0, 1, 2, 3]
        ),
    ),
    (
        "c3ry",
        4,
        "XYYX",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RY(angle).control(3), [0, 1, 2, 3]
        ),
    ),
    (
        "c3rz",
        4,
        "ZXYX",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.RZ(angle).control(3), [0, 1, 2, 3]
        ),
    ),
    (
        "c3phase",
        4,
        "XXYX",
        lambda circuit, angle: circuit.append_mc_gate(
            StandardGate.Phase(angle).control(3), [0, 1, 2, 3]
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


class TestAnalyticValues:
    def test_single_ry_matches_negative_sine(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",)
        )
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx((-0.3894183423086505,), abs=1e-10)

    def test_shared_symbol_sums_over_occurrences(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.3}, ("theta",)
        )
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        reference = ParameterShiftGradient().run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx(reference.values, abs=1e-10)

    def test_hamiltonian_observable_is_extended(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        observable = Hamiltonian.from_list(
            [(PauliString.from_str("Z"), 0.5), (PauliString.from_str("X"), 0.25)]
        )
        request = _request(circuit, observable, {"theta": 0.4}, ("theta",))
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        expected = -0.5 * np.sin(0.4) + 0.25 * np.cos(0.4)
        assert result.values == pytest.approx((expected,), abs=1e-10)

    def test_negative_pauli_phase_is_preserved(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        observable = PauliString.from_str("Z")
        observable.phase = Phase.minus()
        request = _request(circuit, observable, {"theta": 0.4}, ("theta",))
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx((np.sin(0.4),), abs=1e-10)

    def test_hamiltonian_term_phase_is_preserved(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        phased = PauliString.from_str("Z")
        phased.phase = Phase.i()
        observable = Hamiltonian.from_list([(phased, -1j)])
        bindings = {"theta": 0.4}
        request = _request(circuit, observable, bindings, ("theta",))
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        expected = _central_difference(
            matrix_estimator, circuit, observable, bindings, "theta"
        )
        assert result.values == pytest.approx((expected,), abs=1e-7)

    def test_affine_expression_applies_chain_rule(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, 2 * theta + 1)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.25}, ("theta",)
        )
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        expected = -2.0 * np.sin(1.5)
        assert result.values == pytest.approx((expected,), abs=1e-10)

    def test_u_gate_three_slots(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lam")
        circuit = Circuit(1)
        circuit.h(0)
        circuit.u(0, theta, phi, lam)
        request = _request(
            circuit,
            PauliString.from_str("X"),
            {"theta": 0.4, "phi": 0.2, "lam": 0.6},
            ("theta", "phi", "lam"),
        )
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        reference = ParameterShiftGradient().run(matrix_estimator, [request])[0]
        assert result.values == pytest.approx(reference.values, abs=1e-9)
        for name, value in zip(("theta", "phi", "lam"), result.values, strict=True):
            expected = _central_difference(
                matrix_estimator,
                circuit,
                PauliString.from_str("X"),
                {"theta": 0.4, "phi": 0.2, "lam": 0.6},
                name,
            )
            assert value == pytest.approx(expected, abs=1e-7)


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
            circuit, PauliString.from_str(pauli), {"theta": 0.66}, ("theta",)
        )
        lc = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        ps = ParameterShiftGradient().run(matrix_estimator, [request])[0]
        assert lc.values == pytest.approx(ps.values, abs=1e-9)

    def test_matches_independent_finite_difference(
        self, matrix_estimator, gate_name, num_qubits, pauli, append_gate
    ) -> None:
        theta = Parameter("theta")
        circuit = _gate_family_circuit(num_qubits, append_gate, theta)
        observable = PauliString.from_str(pauli)
        bindings = {"theta": 0.41}
        request = _request(circuit, observable, bindings, ("theta",))
        result = LinCombEstimatorGradient().run(matrix_estimator, [request])[0]
        expected = _central_difference(
            matrix_estimator, circuit, observable, bindings, "theta"
        )
        assert abs(expected) > 1e-6, gate_name
        assert result.values == pytest.approx((expected,), abs=1e-7)


class TestCircuitShape:
    def test_auxiliary_qubit_is_appended(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.4}, ("theta",)
        )
        seen = {}

        class Inspecting(matrix_estimator.__class__):
            def run(self, requests, /):
                for request in requests:
                    seen[request.circuit.num_qubits] = True
                    break
                return super().run(requests)

        LinCombEstimatorGradient().run(Inspecting(), [request])
        assert seen == {2: True}

    def test_fresh_auxiliary_respects_reordered_non_contiguous_ids(
        self, matrix_estimator
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit([Qubit(5), Qubit(2)])
        circuit.h(circuit.qubits[1])
        circuit.cry(circuit.qubits[1], circuit.qubits[0], theta)
        observable = PauliString.from_str("XY")
        bindings = {"theta": 0.37}
        request = _request(circuit, observable, bindings, ("theta",))
        observed_orders = []

        class Inspecting(matrix_estimator.__class__):
            def run(self, requests, /):
                requests = list(requests)
                observed_orders.extend(
                    [qubit.index for qubit in item.circuit.qubits] for item in requests
                )
                return super().run(requests)

        result = LinCombEstimatorGradient().run(Inspecting(), [request])[0]
        expected = _central_difference(
            matrix_estimator, circuit, observable, bindings, "theta"
        )
        assert observed_orders == [[5, 2, 0], [5, 2, 0]]
        assert result.values == pytest.approx((expected,), abs=1e-7)

    def test_one_evaluation_per_occurrence(self, recording_estimator_factory) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.3}, ("theta",)
        )
        estimator = recording_estimator_factory()
        LinCombEstimatorGradient().run(estimator, [request])
        assert estimator.batch_sizes == [2]

    @pytest.mark.parametrize(
        ("control_count", "expected_evaluations"),
        [(1, 2), (2, 4), (3, 8)],
    )
    def test_controlled_rotation_term_count(
        self, recording_estimator_factory, control_count, expected_evaluations
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit(control_count + 1)
        circuit.append_mc_gate(
            StandardGate.RX(theta).control(control_count),
            list(range(control_count + 1)),
        )
        observable = PauliString.from_str("X" * control_count + "Y")
        request = _request(circuit, observable, {"theta": 0.3}, ("theta",))
        estimator = recording_estimator_factory()
        LinCombEstimatorGradient().run(estimator, [request])
        assert estimator.batch_sizes == [expected_evaluations]

    def test_single_flattened_batch_across_requests(
        self, recording_estimator_factory
    ) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        first_circuit = Circuit(1)
        first_circuit.ry(0, theta)
        second_circuit = Circuit(1)
        second_circuit.rx(0, phi)
        first = _request(
            first_circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",)
        )
        second = _request(
            second_circuit, PauliString.from_str("Z"), {"phi": -0.4}, ("phi",)
        )
        estimator = recording_estimator_factory()
        results = LinCombEstimatorGradient().run(estimator, [first, second])
        assert estimator.batch_sizes == [2]
        assert len(results) == 2

    def test_empty_parameter_names_skip_estimator(
        self, constant_estimator_factory
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.2}, ())
        estimator = constant_estimator_factory(1.0)
        result = LinCombEstimatorGradient().run(estimator, [request])[0]
        assert result.parameter_names == ()
        assert result.values == ()
        assert estimator.calls == 0

    def test_global_phase_parameter_needs_no_evaluation(
        self, recording_estimator_factory
    ) -> None:
        gphase = Parameter("gphase")
        circuit = Circuit(1)
        circuit.ry(0, 0.3)
        circuit.append_gate(StandardGate.GPhase(gphase), [])
        request = _request(
            circuit,
            PauliString.from_str("Z"),
            {"gphase": 0.9},
            ("gphase",),
        )
        estimator = recording_estimator_factory()
        result = LinCombEstimatorGradient().run(estimator, [request])[0]
        assert result.values == (0.0,)
        assert estimator.batch_sizes == []

    def test_circuit_global_phase_needs_no_evaluation(
        self, recording_estimator_factory
    ) -> None:
        phase = Parameter("phase")
        circuit = Circuit(1)
        circuit.ry(0, 0.3)
        circuit.set_global_phase(phase)
        request = _request(
            circuit, PauliString.from_str("Z"), {"phase": 0.7}, ("phase",)
        )
        estimator = recording_estimator_factory()
        result = LinCombEstimatorGradient().run(estimator, [request])[0]
        assert result.values == (0.0,)
        assert estimator.batch_sizes == []

    def test_empty_request_batch_skips_estimator(
        self, constant_estimator_factory
    ) -> None:
        estimator = constant_estimator_factory(1.0)
        assert LinCombEstimatorGradient().run(estimator, []) == []
        assert estimator.calls == 0

    def test_result_order_follows_request_order(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        first_circuit = Circuit(1)
        first_circuit.ry(0, theta)
        second_circuit = Circuit(1)
        second_circuit.rx(0, phi)
        requests = [
            _request(
                first_circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",)
            ),
            _request(
                second_circuit, PauliString.from_str("Z"), {"phi": -0.4}, ("phi",)
            ),
        ]
        results = LinCombEstimatorGradient().run(matrix_estimator, requests)
        assert results[0].parameter_names == ("theta",)
        assert results[1].parameter_names == ("phi",)


class TestValidation:
    def test_estimator_type_is_checked(self, matrix_estimator) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",)
        )
        with pytest.raises(TypeError, match="Estimator"):
            LinCombEstimatorGradient().run(object(), [request])  # type: ignore[arg-type]

    def test_request_type_is_checked(self, matrix_estimator) -> None:
        with pytest.raises(TypeError, match="GradientRequest"):
            LinCombEstimatorGradient().run(matrix_estimator, [object()])  # type: ignore[list-item]

    def test_unsupported_gate_raises_before_estimator(
        self, constant_estimator_factory
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit(2)
        circuit.ry(0, 0.2)
        circuit.ry(1, 0.3)
        circuit.xy(1, theta)
        request = _request(
            circuit, PauliString.from_str("ZY"), {"theta": 0.5}, ("theta",)
        )
        estimator = constant_estimator_factory(1.0)
        with pytest.raises(ValueError, match="not supported"):
            LinCombEstimatorGradient().run(estimator, [request])
        assert estimator.calls == 0

    def test_estimator_result_count_mismatch(
        self, constant_estimator_factory, exact_result_factory
    ) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.2}, ("theta",)
        )

        class Broken(constant_estimator_factory):
            def run(self, requests, /):
                self.calls += 1
                return []

        with pytest.raises(ValueError, match="per evaluation"):
            LinCombEstimatorGradient().run(Broken(1.0), [request])
