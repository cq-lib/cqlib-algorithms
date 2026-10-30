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

"""Quantum Fisher information tests against the QGT metric."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit.gates import StandardGate
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import (
    NaturalGradient,
    ParameterShiftGradient,
    QFIResult,
    StatevectorQFI,
    StatevectorQGT,
)
from cqlib_algorithms.protocols import EstimatorRequest, GradientRequest


def _request(circuit, observable, bindings, names):
    return GradientRequest(EstimatorRequest(circuit, observable, bindings), names)


class TestQFIResultValidation:
    def test_accepts_valid_matrix(self) -> None:
        result = QFIResult(("a",), np.array([[1.0]]))
        assert result.parameter_names == ("a",)
        assert result.qfi.shape == (1, 1)

    def test_rejects_non_tuple_names(self) -> None:
        with pytest.raises(TypeError, match="tuple"):
            QFIResult(["a"], np.array([[1.0]]))  # type: ignore[arg-type]

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            QFIResult(("a", "a"), np.eye(2))

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="square"):
            QFIResult(("a",), np.ones((2, 2)))

    def test_rejects_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            QFIResult(("a",), np.array([[np.nan]]))

    def test_rejects_asymmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            QFIResult(("a", "b"), np.array([[1.0, 2.0], [0.0, 1.0]]))

    def test_rejects_complex_with_imaginary_part(self) -> None:
        with pytest.raises(ValueError, match="real"):
            QFIResult(("a",), np.array([[1.0 + 1j]]))


class TestAnalyticValues:
    def test_single_ry_qfi_is_one(self) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(
            circuit, PauliString.from_str("Z"), {"theta": 0.3}, ("theta",)
        )
        result = StatevectorQFI().compute(request)
        assert result.parameter_names == ("theta",)
        assert result.qfi[0, 0] == pytest.approx(1.0, abs=1e-10)

    def test_two_independent_rotations_are_diagonal(self) -> None:
        first = Parameter("a")
        second = Parameter("b")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.ry(1, second)
        request = _request(
            circuit,
            PauliString.from_str("ZZ"),
            {"a": 0.2, "b": 0.9},
            ("a", "b"),
        )
        result = StatevectorQFI().compute(request)
        np.testing.assert_allclose(result.qfi, np.eye(2), atol=1e-10)

    def test_empty_parameters_give_empty_matrix(self) -> None:
        theta = Parameter("theta")
        circuit = Circuit(1)
        circuit.ry(0, theta)
        request = _request(circuit, PauliString.from_str("Z"), {"theta": 0.3}, ())
        result = StatevectorQFI().compute(request)
        assert result.parameter_names == ()
        assert result.qfi.shape == (0, 0)

    def test_global_phase_parameter_has_zero_qfi(self) -> None:
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
        result = StatevectorQFI().compute(request)
        assert result.qfi[0, 0] == pytest.approx(0.0, abs=1e-10)


class TestQGTConsistency:
    def test_qfi_is_four_times_metric(self) -> None:
        first = Parameter("a")
        second = Parameter("b")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.rz(1, second)
        circuit.cx(0, 1)
        circuit.ry(0, 0.4)
        request = _request(
            circuit,
            PauliString.from_str("YZ"),
            {"a": 0.5, "b": -0.2},
            ("a", "b"),
        )
        qgt = StatevectorQGT().compute(request)
        qfi = StatevectorQFI().compute(request)
        np.testing.assert_allclose(qfi.qfi, 4.0 * qgt.metric, atol=1e-12)

    def test_request_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="GradientRequest"):
            StatevectorQFI().compute(object())  # type: ignore[arg-type]


class TestNaturalGradientRelation:
    def test_direction_is_invariant_under_metric_scaling(
        self, matrix_estimator
    ) -> None:
        first = Parameter("a")
        second = Parameter("b")
        circuit = Circuit(2)
        circuit.ry(0, first)
        circuit.cx(0, 1)
        circuit.rz(1, second)
        request = _request(
            circuit,
            PauliString.from_str("ZI"),
            {"a": 0.6, "b": 0.1},
            ("a", "b"),
        )
        qgt = StatevectorQGT().compute(request)
        natural = NaturalGradient().run(matrix_estimator, [request])
        direction = np.array(natural[0].values)
        analytic = ParameterShiftGradient().run(matrix_estimator, [request])
        gradient = np.array(analytic[0].values)
        # NaturalGradient solves g * eta = grad; using the QFI (4g) as the
        # metric must give the same direction, scaled by 1/4.
        scaled_eta = np.linalg.lstsq(4.0 * qgt.metric, gradient, rcond=None)[0]
        np.testing.assert_allclose(4.0 * scaled_eta, direction, atol=1e-9)
