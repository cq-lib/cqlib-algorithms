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

"""Tests for the single-qubit quantum signal processing component."""

from __future__ import annotations

import inspect
import math
from dataclasses import is_dataclass

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval

from cqlib import Circuit, Qubit
from cqlib_algorithms.operators.oracle import QuantumSignalProcessing


def _signal_matrix(signal_value: float) -> np.ndarray:
    amplitude = math.sqrt(max(0.0, 1.0 - signal_value * signal_value))
    return np.asarray(
        [
            [signal_value, 1j * amplitude],
            [1j * amplitude, signal_value],
        ],
        dtype=np.complex128,
    )


def _phase_matrix(phase: float) -> np.ndarray:
    return np.asarray(
        [
            [np.exp(1j * phase), 0.0],
            [0.0, np.exp(-1j * phase)],
        ],
        dtype=np.complex128,
    )


def _qsp_reference(signal_value: float, phases: tuple[float, ...]) -> np.ndarray:
    matrix = _phase_matrix(phases[0])
    signal = _signal_matrix(signal_value)
    for phase in phases[1:]:
        matrix = matrix @ signal @ _phase_matrix(phase)
    return matrix


def _assert_su2_structure(matrix: np.ndarray) -> None:
    identity = np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0 + 0.0j, abs=1e-12)
    assert matrix[1, 1] == pytest.approx(np.conjugate(matrix[0, 0]), abs=1e-12)
    assert matrix[1, 0] == pytest.approx(-np.conjugate(matrix[0, 1]), abs=1e-12)


def test_build_circuit_matches_independent_su2_reference() -> None:
    operator = QuantumSignalProcessing(
        [0.5768, -0.1132, -0.1132, 0.5768],
        signal_value=0.3,
    )

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        _qsp_reference(operator.signal_value, operator.phases),
        rtol=0.0,
        atol=1e-12,
    )


def test_built_qsp_matrix_has_expected_su2_structure() -> None:
    matrix = QuantumSignalProcessing(
        [0.4, -0.2, 0.1],
        signal_value=0.3,
    ).build_circuit().to_matrix()

    _assert_su2_structure(matrix)


def test_signal_angle_and_signal_value_define_the_same_unitary() -> None:
    angle = math.acos(0.25)
    phases = [0.4, -0.2, 0.1]

    from_value = QuantumSignalProcessing(phases, signal_value=0.25)
    from_angle = QuantumSignalProcessing(phases, signal_angle=angle)

    np.testing.assert_allclose(
        from_value.build_circuit().to_matrix(),
        from_angle.build_circuit().to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


def test_high_degree_zero_phase_sequence_realizes_chebyshev_polynomial() -> None:
    degree = 20
    phases = [0.0] * (degree + 1)
    xs = np.linspace(-1.0, 1.0, 101, dtype=np.float64)

    approximated = np.asarray(
        [
            np.real(
                QuantumSignalProcessing(phases, signal_value=float(x))
                .build_circuit()
                .to_matrix()[0, 0]
            )
            for x in xs
        ],
        dtype=np.float64,
    )
    expected = np.cos(degree * np.arccos(xs))

    np.testing.assert_allclose(approximated, expected, rtol=0.0, atol=1e-12)


def test_build_returns_fresh_independent_circuits() -> None:
    operator = QuantumSignalProcessing([0.4, -0.2, 0.1], signal_value=0.3)
    first = operator.build_circuit()
    second = operator.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_qsp_circuit_composes_onto_non_contiguous_qubits() -> None:
    operator = QuantumSignalProcessing([0.4, -0.2, 0.1], signal_value=0.3)
    destination = Circuit([7])
    source = operator.build_circuit()
    destination.compose(source, [Qubit(7)])

    np.testing.assert_allclose(
        destination.to_matrix([7]),
        source.to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


def test_build_circuit_uses_expected_gate_sequence_and_angles() -> None:
    phases = [0.4, -0.2, 0.1, 0.3]
    theta = math.acos(0.25)
    circuit = QuantumSignalProcessing(phases, signal_value=0.25).build_circuit()

    assert [operation.name for operation in circuit.operations] == [
        "RZ",
        "RX",
        "RZ",
        "RX",
        "RZ",
        "RX",
        "RZ",
    ]
    assert [list(operation.qubits)[0].index for operation in circuit.operations] == [
        0
    ] * 7
    assert [
        float(operation.params[0]) for operation in circuit.operations
    ] == pytest.approx(
        [
            -2.0 * phases[0],
            -2.0 * theta,
            -2.0 * phases[1],
            -2.0 * theta,
            -2.0 * phases[2],
            -2.0 * theta,
            -2.0 * phases[3],
        ]
    )


def test_qsp_component_is_slotted_and_keyword_only() -> None:
    signature = inspect.signature(QuantumSignalProcessing)
    operator = QuantumSignalProcessing([0.1], signal_value=0.2)

    assert signature.parameters["signal_value"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["signal_angle"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not is_dataclass(operator)
    assert not hasattr(operator, "__dict__")
    assert operator.num_qubits == 1
    with pytest.raises(TypeError):
        QuantumSignalProcessing([0.1], 0.2)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        operator.phases = (0.2,)  # type: ignore[misc]


@pytest.mark.parametrize("phases", [[], np.asarray([])])
def test_qsp_rejects_empty_phase_sequences(phases) -> None:
    with pytest.raises(ValueError, match="phases must not be empty"):
        QuantumSignalProcessing(phases, signal_value=0.1)


@pytest.mark.parametrize("phases", [[True], ["0.1"], [math.nan]])
def test_qsp_rejects_non_real_finite_phase_entries(phases) -> None:
    with pytest.raises(TypeError, match="phases must contain real finite scalars"):
        QuantumSignalProcessing(phases, signal_value=0.1)  # type: ignore[arg-type]


def test_qsp_requires_exactly_one_signal_specification() -> None:
    with pytest.raises(
        ValueError,
        match="exactly one of signal_value and signal_angle",
    ):
        QuantumSignalProcessing([0.1])
    with pytest.raises(
        ValueError,
        match="exactly one of signal_value and signal_angle",
    ):
        QuantumSignalProcessing([0.1], signal_value=0.2, signal_angle=0.3)


@pytest.mark.parametrize("signal_value", [True, "0.1", complex(0.1, 0.0), math.inf])
def test_qsp_rejects_non_real_finite_signal_value(signal_value) -> None:
    with pytest.raises(TypeError, match="signal_value must be a real finite scalar"):
        QuantumSignalProcessing(
            [0.1],
            signal_value=signal_value,
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("signal_value", [-1.1, 1.1])
def test_qsp_rejects_signal_values_outside_closed_interval(signal_value: float) -> None:
    with pytest.raises(ValueError, match=r"-1 <= signal_value <= 1"):
        QuantumSignalProcessing([0.1], signal_value=signal_value)


@pytest.mark.parametrize("signal_angle", [True, "0.1", complex(0.1, 0.0), math.nan])
def test_qsp_rejects_non_real_finite_signal_angle(signal_angle) -> None:
    with pytest.raises(TypeError, match="signal_angle must be a real finite scalar"):
        QuantumSignalProcessing(
            [0.1],
            signal_angle=signal_angle,
        )  # type: ignore[arg-type]


@pytest.mark.parametrize("signal_value", [-1.0, 1.0])
def test_qsp_matches_reference_at_closed_interval_endpoints(
    signal_value: float,
) -> None:
    operator = QuantumSignalProcessing([0.4, -0.2, 0.1], signal_value=signal_value)
    matrix = operator.build_circuit().to_matrix()

    np.testing.assert_allclose(
        matrix,
        _qsp_reference(signal_value, operator.phases),
        rtol=0.0,
        atol=1e-12,
    )
    _assert_su2_structure(matrix)


@pytest.mark.parametrize("signal_value", [-1.0 + 1e-12, 1.0 - 1e-12])
def test_qsp_remains_finite_and_unitary_near_interval_endpoints(
    signal_value: float,
) -> None:
    matrix = QuantumSignalProcessing(
        [0.4, -0.2, 0.1],
        signal_value=signal_value,
    ).build_circuit().to_matrix()

    assert np.all(np.isfinite(matrix))
    _assert_su2_structure(matrix)


def test_solve_phases_matches_known_degree_three_reference_polynomial() -> None:
    coefficients = [0.0, 0.2, 0.0, 0.4]
    phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    xs = np.linspace(-1.0, 1.0, 51, dtype=np.float64)
    approximated = np.asarray(
        [
            np.real(
                _qsp_reference(
                    float(x),
                    tuple(float(value) for value in phases),
                )[0, 0]
            )
            for x in xs
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        approximated,
        chebval(xs, coefficients),
        rtol=0.0,
        atol=1e-5,
    )


def test_solve_phases_returns_a_symmetric_phase_sequence() -> None:
    phases = QuantumSignalProcessing.solve_phases(
        [0.0, 0.2, 0.0, 0.4],
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )

    np.testing.assert_allclose(phases, phases[::-1], rtol=0.0, atol=1e-12)


def test_from_chebyshev_coefficients_solves_and_binds_signal() -> None:
    operator = QuantumSignalProcessing.from_chebyshev_coefficients(
        [0.0, 0.5],
        signal_value=0.4,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )

    matrix = operator.build_circuit().to_matrix()
    assert operator.signal_value == pytest.approx(0.4)
    assert len(operator.phases) == 2
    assert np.real(matrix[0, 0]) == pytest.approx(0.5 * 0.4, abs=1e-6)


def test_from_chebyshev_coefficients_accepts_signal_angle() -> None:
    angle = math.acos(0.4)
    operator = QuantumSignalProcessing.from_chebyshev_coefficients(
        [0.0, 0.5],
        signal_angle=angle,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )

    matrix = operator.build_circuit().to_matrix()
    assert operator.signal_angle == pytest.approx(angle)
    assert operator.signal_value == pytest.approx(0.4)
    assert np.real(matrix[0, 0]) == pytest.approx(0.5 * 0.4, abs=1e-6)


def test_solve_phases_handles_degree_zero_exactly() -> None:
    phases = QuantumSignalProcessing.solve_phases([0.3])
    operator = QuantumSignalProcessing(phases, signal_value=-0.8)
    matrix = operator.build_circuit().to_matrix()

    assert phases.shape == (1,)
    assert np.real(matrix[0, 0]) == pytest.approx(0.3, abs=1e-12)


def test_qsp_inverse_matches_conjugate_transpose() -> None:
    circuit = QuantumSignalProcessing(
        [0.4, -0.2, 0.1],
        signal_value=0.3,
    ).build_circuit()

    np.testing.assert_allclose(
        circuit.inverse().to_matrix(),
        circuit.to_matrix().conj().T,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("coefficients", [True, "x", object()])
def test_solve_phases_rejects_non_sequence_coefficients(coefficients) -> None:
    with pytest.raises(
        TypeError,
        match="coefficients must be a one-dimensional real sequence",
    ):
        QuantumSignalProcessing.solve_phases(coefficients)  # type: ignore[arg-type]


@pytest.mark.parametrize("coefficients", [[True], [complex(1.0, 0.0)], [math.nan]])
def test_solve_phases_rejects_non_real_finite_coefficients(coefficients) -> None:
    with pytest.raises(
        TypeError,
        match="coefficients must contain real finite scalars",
    ):
        QuantumSignalProcessing.solve_phases(coefficients)  # type: ignore[arg-type]


def test_solve_phases_rejects_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="coefficients must not be empty"):
        QuantumSignalProcessing.solve_phases([])


def test_solve_phases_rejects_mismatched_chebyshev_parity() -> None:
    with pytest.raises(ValueError, match="parity matching the degree"):
        QuantumSignalProcessing.solve_phases([1.0, 1.0])


def test_solve_phases_rejects_polynomials_outside_unit_ball() -> None:
    with pytest.raises(ValueError, match=r"\|f\(x\)\| <= 1"):
        QuantumSignalProcessing.solve_phases([1.2])


def test_solve_phases_raises_runtime_error_when_iteration_budget_is_too_small() -> None:
    with pytest.raises(RuntimeError, match="did not converge within max_iterations"):
        QuantumSignalProcessing.solve_phases(
            [0.0, 0.2, 0.0, 0.4],
            tolerance=1e-16,
            max_iterations=1,
            gradient_tolerance=1e-16,
        )
