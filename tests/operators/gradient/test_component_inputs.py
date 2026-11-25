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

"""Ansatz-driven derivatives and geometry against analytic references and batches."""

import math

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators import RealAmplitudesAnsatz, UniformSuperposition
from cqlib_algorithms.operators.gradient import (
    FiniteDifferenceGradient,
    LinCombEstimatorGradient,
    NaturalGradient,
    ParameterShiftGradient,
    SPSAGradient,
    StatevectorQFI,
    StatevectorQGT,
)
from cqlib_algorithms.protocols import (
    Ansatz,
    EstimatorRequest,
    Gradient,
    GradientRequest,
    GradientResult,
)


class _CachedAnsatz(Ansatz):
    def __init__(self, circuit, *, width=None, slots=None):
        self.circuit = circuit
        self.width = circuit.num_qubits if width is None else width
        self.slots = len(circuit.used_symbols) if slots is None else slots
        self.builds = 0

    def validate(self):
        pass

    def num_qubits(self):
        return self.width

    def num_parameters(self):
        return self.slots

    def build_circuit(self, prefix):
        self.builds += 1
        return self.circuit


_GRADIENTS = [
    ParameterShiftGradient,
    LinCombEstimatorGradient,
    FiniteDifferenceGradient,
    lambda: SPSAGradient(1e-5, seed=11),
]


@pytest.mark.parametrize("factory", _GRADIENTS)
def test_ansatz_gradient_matches_analytic_derivative_and_old_batch(
    factory, matrix_estimator
):
    ansatz = RealAmplitudesAnsatz(1, reps=0)
    bindings = {"theta_0": 0.4}
    observable = PauliString.from_str("Z")
    result = factory().compute(matrix_estimator, ansatz, observable, bindings)
    legacy = GradientRequest(
        EstimatorRequest(ansatz.build_circuit("theta"), observable, bindings),
        ("theta_0",),
    )
    assert result == factory().run(matrix_estimator, [legacy])[0]
    assert result.values == pytest.approx((-math.sin(0.4),), abs=1e-8)
    assert bindings == {"theta_0": 0.4}


@pytest.mark.parametrize("factory", _GRADIENTS)
def test_empty_parameter_subset_is_preserved(factory, matrix_estimator):
    result = factory().compute(
        matrix_estimator,
        RealAmplitudesAnsatz(1, reps=0),
        PauliString.from_str("Z"),
        {"theta_0": 0.4},
        parameter_names=(),
    )
    assert result == GradientResult((), ())


def test_single_rotation_geometry_and_natural_direction_match_analytic_values(
    matrix_estimator,
):
    ansatz = RealAmplitudesAnsatz(1, reps=0)
    point = {"angle_0": 0.4}
    qgt = StatevectorQGT().compute(ansatz, point, prefix="angle")
    qfi = StatevectorQFI().compute(ansatz, point, prefix="angle")
    direction = NaturalGradient().compute(
        matrix_estimator, ansatz, PauliString.from_str("Z"), point, prefix="angle"
    )
    np.testing.assert_allclose(qgt.tensor, [[0.25]], atol=1e-14)
    np.testing.assert_allclose(qfi.qfi, [[1.0]], atol=1e-14)
    assert qgt.parameter_names == qfi.parameter_names == ("angle_0",)
    assert direction.values == pytest.approx((-4 * math.sin(0.4),), abs=1e-12)
    regularized = NaturalGradient(0.5).compute(
        matrix_estimator, ansatz, PauliString.from_str("Z"), point, prefix="angle"
    )
    assert regularized.values == pytest.approx((-math.sin(0.4) / 0.75,), abs=1e-12)


def test_complex_qgt_and_reordered_subset_match_independent_state_derivatives(
    matrix_estimator,
):
    theta, phi = 0.6, -0.7
    circuit = Circuit([9])
    circuit.ry(9, Parameter("t"))
    circuit.rz(9, Parameter("p"))
    ansatz = _CachedAnsatz(circuit)
    point = {"t": theta, "p": phi}
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    phase = np.exp(0.5j * phi)
    state = np.array([c / phase, s * phase])
    derivatives = np.array(
        [
            [-0.5j * c / phase, 0.5j * s * phase],
            [-0.5 * s / phase, 0.5 * c * phase],
        ]
    ).T
    expected = (
        derivatives.conj().T @ (np.eye(2) - np.outer(state, state.conj())) @ derivatives
    )
    result = StatevectorQGT().compute(ansatz, point, parameter_names=("p", "t"))
    np.testing.assert_allclose(result.tensor, expected, atol=1e-14)
    np.testing.assert_allclose(
        StatevectorQFI().compute(ansatz, point, parameter_names=("p", "t")).qfi,
        4 * expected.real,
        atol=1e-14,
    )
    gradient = ParameterShiftGradient().compute(
        matrix_estimator,
        ansatz,
        PauliString.from_str("Z"),
        point,
        parameter_names=("p", "t"),
    )
    assert gradient.parameter_names == ("p", "t")
    assert gradient.values == pytest.approx((0, -math.sin(theta)), abs=1e-12)


def test_geometry_component_and_request_paths_agree_and_empty_subset_is_valid():
    ansatz = RealAmplitudesAnsatz(1, reps=0)
    point = {"theta_0": 0.4}
    request = GradientRequest.from_ansatz(ansatz, None, point)
    for component in (StatevectorQGT(), StatevectorQFI()):
        direct = component.compute(ansatz, point)
        legacy = component.compute(request)
        matrix = "tensor" if isinstance(component, StatevectorQGT) else "qfi"
        np.testing.assert_array_equal(getattr(direct, matrix), getattr(legacy, matrix))
        empty = component.compute(ansatz, point, parameter_names=())
        assert getattr(empty, matrix).shape == (0, 0)
        with pytest.raises(ValueError, match="owns"):
            component.compute(request, point)
        with pytest.raises(ValueError, match="requires"):
            component.compute(ansatz)


def test_request_factory_copies_cached_circuit_observable_and_bindings():
    circuit = Circuit(1)
    circuit.ry(0, Parameter("angle"))
    ansatz = _CachedAnsatz(circuit)
    observable = PauliString.from_str("Z")
    bindings = {"angle": 0.4}
    request = GradientRequest.from_ansatz(ansatz, observable, bindings)
    circuit.x(0)
    bindings["angle"] = 1.2
    assert request.estimator_request.circuit is not circuit
    assert request.estimator_request.observable is not observable
    assert len(request.estimator_request.circuit.operations) == 1
    assert request.estimator_request.parameter_bindings["angle"] == 0.4
    assert ansatz.builds == 1


@pytest.mark.parametrize("component", [StatevectorQGT(), StatevectorQFI()])
def test_parameter_free_ansatz_has_empty_geometry(component):
    result = component.compute(_CachedAnsatz(Circuit(1)), {})
    assert result.parameter_names == ()


@pytest.mark.parametrize("kind", ["width", "slots", "return", "reset"])
@pytest.mark.parametrize("surface", ["gradient", "qgt", "qfi", "natural"])
def test_invalid_ansatz_contract_is_rejected(kind, surface, matrix_estimator):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    if kind == "reset":
        circuit.reset(0)
    ansatz = _CachedAnsatz(
        circuit, width=2 if kind == "width" else 1, slots=0 if kind == "slots" else 1
    )
    if kind == "return":
        ansatz.circuit = object()
    with pytest.raises((TypeError, ValueError)):
        if surface == "gradient":
            ParameterShiftGradient().compute(
                matrix_estimator, ansatz, PauliString.from_str("Z"), {"x": 0.4}
            )
        elif surface == "natural":
            NaturalGradient().compute(
                matrix_estimator, ansatz, PauliString.from_str("Z"), {"x": 0.4}
            )
        else:
            component = StatevectorQGT() if surface == "qgt" else StatevectorQFI()
            component.compute(ansatz, {"x": 0.4})


@pytest.mark.parametrize(
    "bindings, options",
    [
        ({}, {}),
        ({"theta_0": math.nan}, {}),
        ({"theta_0": math.inf}, {}),
        ({"theta_0": True}, {}),
        ({"theta_0": 0.4, "extra": 1.0}, {}),
        ({"theta_0": 0.4}, {"parameter_names": ("unknown",)}),
        ({"theta_0": 0.4}, {"parameter_names": ("theta_0", "theta_0")}),
        ({"theta_0": 0.4}, {"prefix": 2}),
        ({"theta_0": 0.4}, {"shots": 0}),
        ({"theta_0": 0.4}, {"seed": -1}),
    ],
)
def test_single_gradient_input_validation(bindings, options, matrix_estimator):
    with pytest.raises((TypeError, ValueError)):
        ParameterShiftGradient().compute(
            matrix_estimator,
            RealAmplitudesAnsatz(1, reps=0),
            PauliString.from_str("Z"),
            bindings,
            **options,
        )


def test_preparation_is_not_silently_used_as_a_trainable_ansatz(matrix_estimator):
    with pytest.raises(TypeError, match="Ansatz"):
        ParameterShiftGradient().compute(
            matrix_estimator, UniformSuperposition(1), PauliString.from_str("Z"), {}
        )


@pytest.mark.parametrize("failure", ["length", "type", "order"])
def test_inherited_single_entry_checks_custom_gradient_response(
    failure, matrix_estimator
):
    class CustomGradient(Gradient):
        def run(self, estimator, requests):
            if failure == "length":
                return []
            if failure == "type":
                return [object()]
            return [GradientResult(("wrong",), (0.0,))]

    with pytest.raises((TypeError, ValueError)):
        CustomGradient().compute(
            matrix_estimator,
            RealAmplitudesAnsatz(1, reps=0),
            PauliString.from_str("Z"),
            {"theta_0": 0.4},
        )


def test_single_entry_passes_shots_seed_and_coordinate_order_to_custom_gradient(
    matrix_estimator,
):
    captured = []

    class CustomGradient(Gradient):
        def run(self, estimator, requests):
            captured.extend(requests)
            return [
                GradientResult(
                    request.parameter_names, (0.0,) * len(request.parameter_names)
                )
                for request in requests
            ]

    result = CustomGradient().compute(
        matrix_estimator,
        RealAmplitudesAnsatz(1, reps=1),
        PauliString.from_str("Z"),
        {"weights_0": 0.2, "weights_1": 0.7},
        prefix="weights",
        parameter_names=("weights_1", "weights_0"),
        shots=777,
        seed=19,
    )
    assert result.parameter_names == ("weights_1", "weights_0")
    assert captured[0].estimator_request.shots == 777
    assert captured[0].estimator_request.seed == 19


@pytest.mark.parametrize("surface", ["gradient", "qgt", "qfi", "natural"])
def test_global_phase_coordinate_does_not_change_observable_or_geometry(
    surface, matrix_estimator
):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("angle"))
    circuit.set_global_phase(Parameter("phase"))
    ansatz = _CachedAnsatz(circuit)
    point = {"angle": 0.4, "phase": 0.8}
    if surface in ("qgt", "qfi"):
        result = (StatevectorQGT() if surface == "qgt" else StatevectorQFI()).compute(
            ansatz, point, parameter_names=("phase",)
        )
        matrix = result.tensor if surface == "qgt" else result.qfi
        np.testing.assert_allclose(matrix, [[0.0]], atol=1e-14)
    else:
        component = (
            ParameterShiftGradient() if surface == "gradient" else NaturalGradient()
        )
        result = component.compute(
            matrix_estimator,
            ansatz,
            PauliString.from_str("Z"),
            point,
            parameter_names=("phase",),
        )
        assert result.values == pytest.approx((0.0,), abs=1e-14)
