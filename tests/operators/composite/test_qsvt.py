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

"""Tests for standard QSVT projector phases and alternating sequences."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from cqlib import Circuit, Qubit
from cqlib.compile import CompilerTransformError
from numpy.polynomial.chebyshev import chebfit, chebval

from cqlib_algorithms.operators.composite import (
    ProjectorControlledPhase,
    QuantumSignalProcessing,
    QuantumSingularValueTransformation,
    RealValuedQSVT,
)
from cqlib_algorithms.operators.composite.quantum_singular_value_transformation._phase_solver import (
    convert_qsp_phases,
    polynomial_to_chebyshev,
)
from cqlib_algorithms.operators.oracle import (
    BlockEncoding,
    UnitaryDilation,
)


class _IdentityStandardBlockEncoding(BlockEncoding):
    def __init__(self) -> None:
        super().__init__(
            alpha=1.0,
            ancilla_qubits=1,
            epsilon=0.0,
            matrix_shape=(2, 2),
            system_qubits=1,
        )

    def build_circuit(self) -> Circuit:
        return Circuit(self.total_qubits)


class _IdentityNonStandardBlockEncoding(_IdentityStandardBlockEncoding):
    @property
    def is_standard(self) -> bool:
        return False


def _projector_phase_matrix(
    num_system_qubits: int,
    num_ancilla_qubits: int,
    phase: float,
) -> np.ndarray:
    system_identity = np.eye(1 << num_system_qubits, dtype=np.complex128)
    if num_ancilla_qubits == 0:
        projector = system_identity
    else:
        ancilla_projector = np.zeros(
            (1 << num_ancilla_qubits, 1 << num_ancilla_qubits),
            dtype=np.complex128,
        )
        ancilla_projector[0, 0] = 1.0
        projector = np.kron(ancilla_projector, system_identity)

    dimension = projector.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    return np.asarray(
        np.exp(-1j * phase) * identity
        + (np.exp(1j * phase) - np.exp(-1j * phase)) * projector,
        dtype=np.complex128,
    )


def _assert_is_diagonal(matrix: np.ndarray, *, atol: float) -> None:
    off_diagonal = np.asarray(matrix - np.diag(np.diag(matrix)), dtype=np.complex128)
    np.testing.assert_allclose(off_diagonal, 0.0, rtol=0.0, atol=atol)


def _project_standard_block(
    circuit: Circuit,
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    matrix = np.asarray(circuit.to_matrix(), dtype=np.complex128)
    return np.asarray(matrix[:rows, :columns], dtype=np.complex128)


def _random_square_matrix(dimension: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension)),
        dtype=np.complex128,
    )


def _normalized_svd(
    matrix: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.asarray(matrix / alpha, dtype=np.complex128)
    left_unitary, singular_values, right_unitary_dagger = np.linalg.svd(
        normalized,
        full_matrices=False,
    )
    return (
        np.asarray(left_unitary, dtype=np.complex128),
        np.asarray(singular_values, dtype=np.float64),
        np.asarray(right_unitary_dagger.conj().T, dtype=np.complex128),
    )


def _evaluate_chebyshev_on_singular_values(
    singular_values: np.ndarray,
    *,
    chebyshev_coefficients: list[float],
) -> np.ndarray:
    return np.asarray(
        [chebval(float(value), chebyshev_coefficients) for value in singular_values],
        dtype=np.float64,
    )


def _qsvt_singular_basis_block(
    projected: np.ndarray,
    *,
    left_unitary: np.ndarray,
    right_unitary: np.ndarray,
    degree: int,
) -> np.ndarray:
    r"""Return the projected block expressed in the SVD basis of the encoding.

    An even target acts as ``V diag(P(sigma)) V^dag`` on the encoding, an odd
    one as ``U diag(P(sigma)) V^dag``.
    """
    if degree % 2 == 0:
        return np.asarray(
            right_unitary.conj().T @ projected @ right_unitary,
            dtype=np.complex128,
        )
    return np.asarray(
        left_unitary.conj().T @ projected @ right_unitary,
        dtype=np.complex128,
    )


def _assert_qsvt_matches_singular_vector_action(
    projected: np.ndarray,
    *,
    left_unitary: np.ndarray,
    right_unitary: np.ndarray,
    expected_values: np.ndarray,
    degree: int,
    atol: float,
) -> None:
    singular_basis_matrix = _qsvt_singular_basis_block(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        degree=degree,
    )

    diagonal = np.asarray(np.diag(singular_basis_matrix), dtype=np.complex128)
    off_diagonal = np.asarray(
        singular_basis_matrix - np.diag(diagonal),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        np.real(diagonal),
        expected_values,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        off_diagonal,
        0.0,
        rtol=0.0,
        atol=atol,
    )


def _assert_qsvt_matches_complex_singular_vector_action(
    projected: np.ndarray,
    *,
    left_unitary: np.ndarray,
    right_unitary: np.ndarray,
    expected_values: np.ndarray,
    degree: int,
    atol: float,
) -> None:
    r"""Assert the block matches ``diag(P(sigma))`` including its imaginary part."""
    singular_basis_matrix = _qsvt_singular_basis_block(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        degree=degree,
    )

    diagonal = np.asarray(np.diag(singular_basis_matrix), dtype=np.complex128)
    off_diagonal = np.asarray(
        singular_basis_matrix - np.diag(diagonal),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        diagonal,
        expected_values,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        off_diagonal,
        0.0,
        rtol=0.0,
        atol=atol,
    )


def _qsp_top_left_real_response(
    signal_value: float,
    *,
    phases: np.ndarray,
) -> float:
    operator = QuantumSignalProcessing(
        phases,
        signal_value=signal_value,
    )
    matrix = np.asarray(operator.build_circuit().to_matrix(), dtype=np.complex128)
    return float(np.real(matrix[0, 0]))


def _report_qsp_solver_error(
    coefficients: list[float],
    *,
    phases: np.ndarray,
    dimension: int,
    degree: int,
) -> None:
    num_training_nodes = (degree + 2) // 2
    indices = np.arange(1, num_training_nodes + 1, dtype=np.float64)
    training_nodes = np.cos(
        (2.0 * indices - 1.0) * math.pi / (2.0 * num_training_nodes),
    )
    training_targets = chebval(training_nodes, coefficients)
    training_predictions = np.asarray(
        [
            _qsp_top_left_real_response(float(node), phases=phases)
            for node in training_nodes
        ],
        dtype=np.float64,
    )
    training_mse = float(
        np.mean(np.square(training_predictions - training_targets)),
    )

    probe_points = np.linspace(-1.0, 1.0, 129, dtype=np.float64)
    probe_targets = chebval(probe_points, coefficients)
    probe_predictions = np.asarray(
        [
            _qsp_top_left_real_response(float(point), phases=phases)
            for point in probe_points
        ],
        dtype=np.float64,
    )
    probe_max_error = float(np.max(np.abs(probe_predictions - probe_targets)))
    print(
        "QSP solver "
        f"(dimension={dimension}, degree={degree}): "
        f"training_mse={training_mse:.3e}, "
        f"probe_max_err={probe_max_error:.3e}"
    )


def _chebyshev_degree_coefficients(degree: int) -> list[float]:
    coefficients = [0.0] * (degree + 1)
    coefficients[degree] = 1.0
    return coefficients


def _assert_qsvt_chebyshev_degree_matches_singular_vector_action(
    *,
    dimension: int,
    degree: int,
    seed: int,
    atol: float,
) -> None:
    matrix = _random_square_matrix(dimension, seed=seed)
    spectral_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    alpha = 2.0 * spectral_norm
    coefficients = _chebyshev_degree_coefficients(degree)
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    _report_qsp_solver_error(
        coefficients,
        phases=qsp_phases,
        dimension=dimension,
        degree=degree,
    )
    encoding = UnitaryDilation(matrix, alpha=alpha, validate=False)
    operator = QuantumSingularValueTransformation(
        encoding,
        convert_qsp_phases(qsp_phases),
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )
    left_unitary, singular_values, right_unitary = _normalized_svd(
        matrix,
        alpha=alpha,
    )
    expected_values = _evaluate_chebyshev_on_singular_values(
        singular_values,
        chebyshev_coefficients=coefficients,
    )

    _assert_qsvt_matches_singular_vector_action(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        expected_values=expected_values,
        degree=degree,
        atol=atol,
    )


def _qsvt_reference(
    block_encoding: BlockEncoding,
    phases: tuple[float, ...],
) -> np.ndarray:
    unitary = np.asarray(
        block_encoding.build_circuit().to_matrix(), dtype=np.complex128
    )
    return _assemble_qsvt_matrix_from_unitary(
        unitary,
        num_system_qubits=block_encoding.system_qubits,
        num_ancilla_qubits=block_encoding.ancilla_qubits,
        phases=phases,
    )


def _assemble_qsvt_matrix_from_unitary(
    unitary: np.ndarray,
    *,
    num_system_qubits: int,
    num_ancilla_qubits: int,
    phases: tuple[float, ...],
) -> np.ndarray:
    left_template = ProjectorControlledPhase(
        num_system_qubits,
        num_ancilla_qubits,
        phase=0.0,
    )
    right_template = ProjectorControlledPhase(
        num_system_qubits,
        num_ancilla_qubits,
        phase=0.0,
    )
    adjoint = np.asarray(unitary.conj().T, dtype=np.complex128)

    matrix = np.eye(unitary.shape[0], dtype=np.complex128)
    phase_count = len(phases)
    for reverse_index, phase in enumerate(reversed(phases), start=1):
        phase_index = phase_count - reverse_index + 1
        if phase_index % 2 == phase_count % 2:
            matrix = (
                left_template.with_phase(phase).build_circuit().to_matrix()
                @ unitary
                @ matrix
            )
        else:
            matrix = (
                right_template.with_phase(phase).build_circuit().to_matrix()
                @ adjoint
                @ matrix
            )
    return np.asarray(matrix, dtype=np.complex128)


def test_projector_controlled_phase_matches_exact_reference_matrix() -> None:
    operator = ProjectorControlledPhase(1, 2, phase=0.37)

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        _projector_phase_matrix(1, 2, 0.37),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_ancilla_qubits", [1, 2, 3, 4])
def test_projector_controlled_phase_matches_expected_diagonal_pattern(
    num_ancilla_qubits: int,
) -> None:
    operator = ProjectorControlledPhase(0, num_ancilla_qubits, phase=0.5)
    matrix = np.asarray(operator.build_circuit().to_matrix(), dtype=np.complex128)

    _assert_is_diagonal(matrix, atol=1e-15)
    expected_diagonal = np.full(1 << num_ancilla_qubits, np.exp(-0.5j), dtype=complex)
    expected_diagonal[0] = np.exp(0.5j)
    np.testing.assert_allclose(
        np.diag(matrix),
        expected_diagonal,
        rtol=0.0,
        atol=1e-12,
    )


def test_projector_controlled_phase_zero_phase_is_identity() -> None:
    operator = ProjectorControlledPhase(2, 3, phase=0.0)

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        np.eye(1 << operator.num_qubits, dtype=np.complex128),
        rtol=0.0,
        atol=1e-15,
    )


def test_projector_controlled_phase_pi_phase_is_negative_identity() -> None:
    operator = ProjectorControlledPhase(1, 2, phase=math.pi)

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        -np.eye(1 << operator.num_qubits, dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


def test_projector_controlled_phase_respects_phase_additivity() -> None:
    first = ProjectorControlledPhase(1, 2, phase=0.3).build_circuit().to_matrix()
    second = ProjectorControlledPhase(1, 2, phase=0.5).build_circuit().to_matrix()
    combined = ProjectorControlledPhase(1, 2, phase=0.8).build_circuit().to_matrix()

    np.testing.assert_allclose(first @ second, combined, rtol=0.0, atol=1e-12)


def test_projector_controlled_phase_handles_zero_ancilla_as_global_phase() -> None:
    operator = ProjectorControlledPhase(2, 0, phase=-0.19)

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        _projector_phase_matrix(2, 0, -0.19),
        rtol=0.0,
        atol=1e-12,
    )


def test_projector_controlled_phase_acts_trivially_on_system_register() -> None:
    operator = ProjectorControlledPhase(2, 2, phase=0.7)
    matrix = np.asarray(operator.build_circuit().to_matrix(), dtype=np.complex128)
    projected_zero_block = matrix[:4, :4]
    projected_nonzero_block = matrix[4:8, 4:8]

    np.testing.assert_allclose(
        projected_zero_block,
        np.exp(0.7j) * np.eye(4, dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        projected_nonzero_block,
        np.exp(-0.7j) * np.eye(4, dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


def test_projector_controlled_phase_exposes_local_register_metadata() -> None:
    signature = inspect.signature(ProjectorControlledPhase)
    operator = ProjectorControlledPhase(2, 3, phase=0.0)

    assert signature.parameters["phase"].kind is inspect.Parameter.KEYWORD_ONLY
    assert operator.num_system_qubits == 2
    assert operator.num_ancilla_qubits == 3
    assert operator.num_qubits == 5
    assert operator.system_register_indices == [0, 1]
    assert operator.ancilla_register_indices == [2, 3, 4]


def test_projector_controlled_phase_build_returns_fresh_circuits() -> None:
    operator = ProjectorControlledPhase(1, 1, phase=0.23)
    first = operator.build_circuit()
    second = operator.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_projector_controlled_phase_composes_onto_non_contiguous_qubits() -> None:
    source = ProjectorControlledPhase(1, 2, phase=0.41).build_circuit()
    destination = Circuit([2, 5, 7])
    destination.compose(source, [Qubit(2), Qubit(5), Qubit(7)])

    np.testing.assert_allclose(
        destination.to_matrix([2, 5, 7]),
        source.to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


def test_projector_controlled_phase_with_phase_returns_a_new_operator() -> None:
    operator = ProjectorControlledPhase(1, 1, phase=0.0)
    shifted = operator.with_phase(math.pi / 7.0)

    assert shifted is not operator
    assert operator.phase == pytest.approx(0.0)
    assert shifted.phase == pytest.approx(math.pi / 7.0)
    np.testing.assert_allclose(
        shifted.build_circuit().to_matrix(),
        _projector_phase_matrix(1, 1, math.pi / 7.0),
        rtol=0.0,
        atol=1e-12,
    )


def test_projector_controlled_phase_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="num_system_qubits must be an integer"):
        ProjectorControlledPhase(True, 1, phase=0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_ancilla_qubits must be an integer"):
        ProjectorControlledPhase(1, 1.5, phase=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        ProjectorControlledPhase(1, -1, phase=0.1)
    with pytest.raises(ValueError, match="total qubit count must be greater than zero"):
        ProjectorControlledPhase(0, 0, phase=0.1)
    with pytest.raises(TypeError, match="phase must be a real finite scalar"):
        ProjectorControlledPhase(1, 1, phase=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="phase must be a real finite scalar"):
        ProjectorControlledPhase(1, 1, phase=np.nan)


def test_qsvt_build_matches_independent_standard_reference() -> None:
    encoding = UnitaryDilation(
        np.array([[0.5, 0.1j], [-0.1j, -0.25]], dtype=np.complex128),
        validate=False,
    )
    operator = QuantumSingularValueTransformation(encoding, [0.3, -0.2, 0.1, 0.05])

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        _qsvt_reference(encoding, operator.phases),
        rtol=0.0,
        atol=1e-12,
    )


def test_scalar_qsvt_matches_independent_standard_reference() -> None:
    encoding = UnitaryDilation(np.array([[0.5]], dtype=np.complex128), validate=False)
    operator = QuantumSingularValueTransformation(encoding, [0.4, -0.2, 0.1])

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        _qsvt_reference(encoding, operator.phases),
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_odd_linear_polynomial_matches_diagonal_svd_reference() -> None:
    matrix = np.diag([0.3, 0.7]).astype(np.complex128)
    encoding = UnitaryDilation(matrix, validate=False)
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        [0.0, 1.0],
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )
    normalized_singular_values = np.diag(matrix) / encoding.alpha
    expected = np.diag(normalized_singular_values)

    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=1e-12)


def test_qsvt_even_constant_polynomial_matches_diagonal_svd_reference() -> None:
    matrix = np.diag([0.3, 0.7]).astype(np.complex128)
    encoding = UnitaryDilation(matrix, validate=False)
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        [1.0],
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )
    expected = np.eye(matrix.shape[0], dtype=np.complex128)

    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=1e-12)


def test_qsvt_degree_zero_identity_matrix_projects_to_identity_block() -> None:
    matrix = np.eye(2, dtype=np.complex128)
    encoding = UnitaryDilation(matrix, validate=False)
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        [1.0],
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )

    np.testing.assert_allclose(
        projected,
        np.eye(2, dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("constant", [0.5, -0.5, 0.0, -1.0])
def test_qsvt_degree_zero_constant_projected_block_matches_complex_completion(
    constant: float,
) -> None:
    matrix = np.diag([0.3, 0.7]).astype(np.complex128)
    encoding = UnitaryDilation(matrix, validate=False)
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        [constant],
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )

    expected_phase = math.acos(constant)
    assert operator.degree == 0
    assert len(operator.phases) == 1
    np.testing.assert_allclose(
        operator.phases[0],
        expected_phase,
        rtol=0.0,
        atol=1e-12,
    )
    expected_block = np.exp(1j * expected_phase) * np.eye(
        matrix.shape[0],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        projected,
        expected_block,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(("dimension", "seed"), [(8, 42), (16, 123)])
@pytest.mark.parametrize("constant", [0.5, -0.5, -1.0])
def test_qsvt_degree_zero_constant_random_matrix_projects_to_scaled_identity(
    dimension: int,
    seed: int,
    constant: float,
) -> None:
    matrix = _random_square_matrix(dimension, seed=seed)
    spectral_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    alpha = 2.0 * spectral_norm
    encoding = UnitaryDilation(matrix, alpha=alpha, validate=False)
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        [constant],
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )

    assert operator.degree == 0
    expected_block = np.exp(1j * math.acos(constant)) * np.eye(
        matrix.shape[0],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        projected,
        expected_block,
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_degree_one_scalar_matches_qsp_response() -> None:
    scalar = 0.4
    coefficients = [0.0, 1.0]
    encoding = UnitaryDilation(
        np.array([[scalar]], dtype=np.complex128), validate=False
    )
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    qsvt = QuantumSingularValueTransformation(
        encoding,
        convert_qsp_phases(qsp_phases),
    )
    qsp = QuantumSignalProcessing(qsp_phases, signal_value=scalar / encoding.alpha)

    projected = _project_standard_block(
        qsvt.build_circuit(),
        rows=1,
        columns=1,
    )
    qsp_top_left = np.asarray(qsp.build_circuit().to_matrix(), dtype=np.complex128)[
        0, 0
    ]

    np.testing.assert_allclose(
        projected[0, 0],
        qsp_top_left,
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_degree_one_scalar_with_explicit_alpha_matches_qsp_response() -> None:
    scalar = 0.4
    alpha = 0.8
    coefficients = [0.0, 1.0]
    encoding = UnitaryDilation(
        np.array([[scalar]], dtype=np.complex128),
        alpha=alpha,
        validate=False,
    )
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    qsvt = QuantumSingularValueTransformation(
        encoding,
        convert_qsp_phases(qsp_phases),
    )
    qsp = QuantumSignalProcessing(qsp_phases, signal_value=scalar / alpha)

    projected = _project_standard_block(
        qsvt.build_circuit(),
        rows=1,
        columns=1,
    )
    qsp_top_left = np.asarray(qsp.build_circuit().to_matrix(), dtype=np.complex128)[
        0, 0
    ]

    np.testing.assert_allclose(
        projected[0, 0],
        qsp_top_left,
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_degree_two_scalar_matches_qsp_response() -> None:
    scalar = 0.4
    coefficients = [0.0, 0.0, 1.0]
    encoding = UnitaryDilation(
        np.array([[scalar]], dtype=np.complex128), validate=False
    )
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    qsvt = QuantumSingularValueTransformation(
        encoding,
        convert_qsp_phases(qsp_phases),
    )
    qsp = QuantumSignalProcessing(qsp_phases, signal_value=scalar / encoding.alpha)

    projected = _project_standard_block(
        qsvt.build_circuit(),
        rows=1,
        columns=1,
    )
    qsp_top_left = np.asarray(qsp.build_circuit().to_matrix(), dtype=np.complex128)[
        0, 0
    ]

    np.testing.assert_allclose(
        projected[0, 0],
        qsp_top_left,
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_high_degree_diagonal_matrix_matches_singular_vector_action() -> None:
    diagonal = np.linspace(0.5, 0.03125, 8, dtype=np.float64)
    matrix = np.diag(diagonal).astype(np.complex128)
    coefficients = _chebyshev_degree_coefficients(10)
    encoding = UnitaryDilation(matrix, alpha=1.0, validate=False)
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=matrix.shape[0],
        columns=matrix.shape[1],
    )
    left_unitary, singular_values, right_unitary = _normalized_svd(
        matrix,
        alpha=encoding.alpha,
    )
    expected_values = _evaluate_chebyshev_on_singular_values(
        singular_values,
        chebyshev_coefficients=coefficients,
    )
    singular_basis_matrix = right_unitary.conj().T @ projected @ right_unitary
    actual_real_diagonal = np.real(np.diag(singular_basis_matrix))
    actual_imag_diagonal = np.imag(np.diag(singular_basis_matrix))
    print("QSVT diagonal diagnostic (degree=10):")
    print("singular_values =", singular_values)
    print("expected_values =", expected_values)
    print("actual_real_diagonal =", actual_real_diagonal)
    print("actual_imag_diagonal =", actual_imag_diagonal)
    print("real_error =", actual_real_diagonal - expected_values)

    _assert_qsvt_matches_singular_vector_action(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        expected_values=expected_values,
        degree=10,
        atol=1e-4,
    )


def test_qsvt_exact_dilation_matrix_matches_circuit_composition_for_high_degree_case() -> (
    None
):
    matrix = _random_square_matrix(8, seed=2024)
    spectral_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    alpha = 2.0 * spectral_norm
    coefficients = _chebyshev_degree_coefficients(8)
    encoding = UnitaryDilation(matrix, alpha=alpha, validate=False)
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    qsvt_phases = convert_qsp_phases(qsp_phases)
    exact_matrix = _assemble_qsvt_matrix_from_unitary(
        encoding.dilation_unitary,
        num_system_qubits=encoding.system_qubits,
        num_ancilla_qubits=encoding.ancilla_qubits,
        phases=qsvt_phases,
    )
    circuit_matrix = _assemble_qsvt_matrix_from_unitary(
        np.asarray(encoding.build_circuit().to_matrix(), dtype=np.complex128),
        num_system_qubits=encoding.system_qubits,
        num_ancilla_qubits=encoding.ancilla_qubits,
        phases=qsvt_phases,
    )
    operator = QuantumSingularValueTransformation(encoding, qsvt_phases)
    built_matrix = np.asarray(operator.build_circuit().to_matrix(), dtype=np.complex128)

    np.testing.assert_allclose(circuit_matrix, exact_matrix, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(built_matrix, exact_matrix, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("dimension", "degree", "seed"),
    [
        (8, 5, 321),
        (16, 7, 707),
        (8, 9, 909),
    ],
)
def test_qsvt_odd_high_degree_polynomials_match_singular_vector_action(
    dimension: int,
    degree: int,
    seed: int,
) -> None:
    _assert_qsvt_chebyshev_degree_matches_singular_vector_action(
        dimension=dimension,
        degree=degree,
        seed=seed,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    ("dimension", "degree", "seed"),
    [
        (8, 6, 654),
        (16, 8, 808),
        (8, 10, 1010),
    ],
)
def test_qsvt_even_high_degree_polynomials_match_singular_vector_action(
    dimension: int,
    degree: int,
    seed: int,
) -> None:
    _assert_qsvt_chebyshev_degree_matches_singular_vector_action(
        dimension=dimension,
        degree=degree,
        seed=seed,
        atol=1e-4,
    )


def test_qsvt_exposes_projector_templates_without_running_block_check() -> None:
    signature = inspect.signature(QuantumSingularValueTransformation)
    encoding = _IdentityStandardBlockEncoding()
    operator = QuantumSingularValueTransformation(encoding, [0.2, -0.1])

    assert (
        signature.parameters["block_encoding"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert operator.block_encoding is encoding
    assert operator.right_projector.phase == pytest.approx(0.0)
    assert operator.left_projector.phase == pytest.approx(0.0)
    assert operator.num_qubits == 2


def test_qsvt_convert_qsp_phases_matches_corollary_formula() -> None:
    qsp_phases = [0.1, -0.2, 0.4, -0.6]
    converted = convert_qsp_phases(qsp_phases)

    np.testing.assert_allclose(
        converted,
        (
            qsp_phases[0] + qsp_phases[-1] + 0.5 * (len(qsp_phases) - 2) * math.pi,
            qsp_phases[1] - 0.5 * math.pi,
            qsp_phases[2] - 0.5 * math.pi,
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_empty_phase_sequence_builds_identity() -> None:
    encoding = UnitaryDilation(
        np.array([[0.2, 0.0], [0.0, -0.1]], dtype=np.complex128),
        validate=False,
    )
    operator = QuantumSingularValueTransformation(encoding, [])

    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        np.eye(1 << operator.num_qubits, dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_build_returns_fresh_independent_circuits() -> None:
    encoding = UnitaryDilation(
        np.array([[0.2, 0.0], [0.0, -0.1]], dtype=np.complex128),
        validate=False,
    )
    operator = QuantumSingularValueTransformation(encoding, [0.4, -0.2, 0.1])
    first = operator.build_circuit()
    second = operator.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_qsvt_circuit_composes_onto_non_contiguous_qubits() -> None:
    encoding = UnitaryDilation(
        np.array([[0.4, 0.0], [0.0, -0.1]], dtype=np.complex128),
        validate=False,
    )
    source = QuantumSingularValueTransformation(
        encoding,
        [0.4, -0.2, 0.1],
    ).build_circuit()
    destination = Circuit([3, 7])
    destination.compose(source, [Qubit(3), Qubit(7)])

    np.testing.assert_allclose(
        destination.to_matrix([3, 7]),
        source.to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_check_block_encoding_uses_explicit_functionality() -> None:
    matrix = np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.complex128)
    encoding = UnitaryDilation(matrix, validate=False)
    operator = QuantumSingularValueTransformation(encoding, [0.1, -0.2])

    assert operator.check_block_encoding() == pytest.approx(0.0, abs=1e-12)
    assert operator.check_block_encoding(expected_matrix=matrix) == pytest.approx(
        0.0,
        abs=1e-12,
    )

    with pytest.raises(
        ValueError,
        match="block encoding does not satisfy A = PiTilde U Pi",
    ):
        operator.check_block_encoding(expected_matrix=np.eye(2, dtype=np.complex128))


def test_qsvt_check_block_encoding_requires_explicit_matrix_for_generic_encoding() -> (
    None
):
    operator = QuantumSingularValueTransformation(
        _IdentityStandardBlockEncoding(),
        [0.1, 0.2],
    )

    with pytest.raises(
        ValueError,
        match=(
            "expected_matrix must be provided when block_encoding does not "
            "expose a matrix attribute"
        ),
    ):
        operator.check_block_encoding()

    assert operator.check_block_encoding(
        expected_matrix=np.eye(2, dtype=np.complex128),
    ) == pytest.approx(0.0, abs=1e-12)


def test_qsvt_from_chebyshev_coefficients_converts_qsp_phase_solver_output() -> None:
    encoding = UnitaryDilation(
        np.array([[0.25, 0.0], [0.0, -0.1]], dtype=np.complex128),
        validate=False,
    )
    expected_phases = QuantumSignalProcessing.solve_phases(
        [0.0, 0.5],
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    operator = QuantumSingularValueTransformation.from_chebyshev_coefficients(
        encoding,
        [0.0, 0.5],
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )

    np.testing.assert_allclose(
        operator.phases,
        convert_qsp_phases(expected_phases),
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_rejects_non_standard_block_encodings_and_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="block_encoding must be a BlockEncoding"):
        QuantumSingularValueTransformation(object(), [0.1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="supports only standard block encodings"):
        QuantumSingularValueTransformation(_IdentityNonStandardBlockEncoding(), [0.1])
    with pytest.raises(TypeError, match="phases must contain real finite scalars"):
        QuantumSingularValueTransformation(
            _IdentityStandardBlockEncoding(),
            [True],
        )  # type: ignore[list-item]


def test_qsvt_check_block_encoding_rejects_invalid_expected_matrix_and_tolerance() -> (
    None
):
    operator = QuantumSingularValueTransformation(
        _IdentityStandardBlockEncoding(),
        [0.1],
    )

    with pytest.raises(TypeError, match="expected_matrix must be a numpy.ndarray"):
        operator.check_block_encoding(
            expected_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="expected_matrix shape must match block_encoding.matrix_shape",
    ):
        operator.check_block_encoding(expected_matrix=np.eye(4, dtype=np.complex128))
    with pytest.raises(
        ValueError,
        match="expected_matrix must contain only finite entries",
    ):
        operator.check_block_encoding(
            expected_matrix=np.array(
                [[1.0, np.inf], [0.0, 1.0]],
                dtype=np.complex128,
            ),
        )
    with pytest.raises(TypeError, match="atol must be a real finite scalar"):
        operator.check_block_encoding(
            expected_matrix=np.eye(2, dtype=np.complex128),
            atol=True,
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="atol must be greater than or equal to zero"):
        operator.check_block_encoding(
            expected_matrix=np.eye(2, dtype=np.complex128),
            atol=-1e-10,
        )


_QSVT_ENCODING_SEED = 20261019
r"""Base seed of the block encodings used by the complex-backend tests."""

_QSVT_ENCODINGS: list[tuple[str, int]] = [
    (family, qubits) for family in ("random", "spectrum") for qubits in (2, 3, 4)
]
r"""Encoding families and sizes of the complex-backend reconstruction tests."""

_QSVT_COMPLEX_BACKENDS: list[str] = ["bfgs", "haah"]
r"""Phase-synthesis backends covered by the complex-backend QSVT tests."""

_QSVT_COMPLEX_BACKEND_ATOL: dict[str, float] = {"bfgs": 1e-5, "haah": 1e-6}
r"""Per-backend block-reconstruction tolerance of the complex QSVT tests."""

_QSVT_COMPLEX_TARGETS: dict[str, np.ndarray] = {
    "iT1": np.array([0.0, 1.0j], dtype=np.complex128),
    "iT2": np.array([0.0, 0.0, 1.0j], dtype=np.complex128),
    "iT3": np.array([0.0, 0.0, 0.0, 1.0j], dtype=np.complex128),
    "iT4": np.array([0.0, 0.0, 0.0, 0.0, 1.0j], dtype=np.complex128),
    "T1": np.array([0.0, 1.0], dtype=np.complex128),
    "T2": np.array([0.0, 0.0, 1.0], dtype=np.complex128),
    "T3": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128),
    "T4": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.complex128),
    "composite-d3": np.array(
        [0.0, 0.5 * np.exp(0.25j * math.pi), 0.0, 0.5 * np.exp(0.25j * math.pi)],
        dtype=np.complex128,
    ),
}
r"""Complex Chebyshev targets satisfying Corollary 8 exactly.

The first eight mirror ``_STRUCTURED_QSP_TARGETS`` of
``test_quantum_signal_processing`` so that both layers share the same
structured coverage.
"""

_QSVT_COMPLEX_RANDOM_CASES: list[tuple[str, int, int, int]] = [
    (family, qubits, degree, seed)
    for family in ("random", "spectrum")
    for qubits in (2, 3, 4)
    for degree in (3, 5, 7)
    for seed in (0, 1)
]
r"""Encoding family, size, degree and seed of the random feasible cases."""

_QSVT_REJECTED_COMPLEX_TARGETS: dict[str, tuple[list[complex], str]] = {
    "degree-0": ([0.5], "Corollary 8 condition 3"),
    "mixed-parity": ([0.2, 0.3, 0.1], "Corollary 8 condition 1"),
    "endpoint-not-normalised": ([0.0, 0.5], "Corollary 8 condition 3"),
}
r"""Infeasible complex targets and the Corollary 8 message each one triggers."""


def _qsvt_encoding(family: str, *, qubits: int) -> UnitaryDilation:
    r"""Return a deterministic encoding used by the complex-backend tests.

    ``"random"`` is a dense complex matrix scaled by twice its spectral norm, so
    its singular values lie in ``(0, 1/2]``; ``"spectrum"`` is a random unitary
    scaled by an explicit diagonal spectrum in ``[0.1, 0.95]``.  Both stay
    generic enough for the recursive unitary synthesis of the encoding.
    """
    size = 1 << qubits
    if family == "random":
        matrix = _random_square_matrix(size, seed=_QSVT_ENCODING_SEED + qubits)
        spectral_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
        return UnitaryDilation(matrix, alpha=2.0 * spectral_norm, validate=False)
    if family == "spectrum":
        rng = np.random.default_rng(_QSVT_ENCODING_SEED + qubits)
        spectrum = rng.uniform(0.1, 0.95, size=size)
        mixing, _ = np.linalg.qr(
            rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        )
        return UnitaryDilation(
            np.asarray(mixing @ np.diag(spectrum), dtype=np.complex128),
            alpha=1.0,
            validate=False,
        )
    raise ValueError(f"unknown encoding family: {family}")


def _qsp_top_left_complex_response(
    signal_value: float,
    *,
    phases: np.ndarray,
) -> complex:
    r"""Return the top-left entry of the R-form QSP response at ``signal_value``."""
    operator = QuantumSignalProcessing(
        phases,
        signal_value=signal_value,
    )
    matrix = np.asarray(operator.build_circuit().to_matrix(), dtype=np.complex128)
    return complex(matrix[0, 0])


def _random_feasible_target(degree: int, *, seed: int) -> np.ndarray:
    r"""Return Chebyshev coefficients of a random Corollary 8 feasible target.

    The response of a random R-form phase sequence is read off at the
    Chebyshev-Lobatto nodes, so the fitted polynomial obeys the QSP unitarity
    relation and hence all four Corollary 8 conditions by construction.
    """
    rng = np.random.default_rng(seed)
    phases = np.asarray(
        rng.uniform(-math.pi, math.pi, size=degree + 1),
        dtype=np.float64,
    )
    nodes = np.cos(math.pi * np.arange(64, dtype=np.float64) / 63.0)
    responses = np.asarray(
        [_qsp_top_left_complex_response(float(node), phases=phases) for node in nodes],
        dtype=np.complex128,
    )
    real_coefficients = chebfit(nodes, responses.real, degree)
    imaginary_coefficients = chebfit(nodes, responses.imag, degree)
    return np.asarray(
        real_coefficients + 1.0j * imaginary_coefficients,
        dtype=np.complex128,
    )


def _assert_qsvt_matches_complex_target(
    encoding: UnitaryDilation,
    coefficients: np.ndarray,
    *,
    synthesis: str,
    atol: float,
) -> None:
    r"""Assert the synthesised block acts as ``P`` on the singular values.

    The QSP phases are synthesised through ``synthesis`` and the complex
    diagonal of the projected block, expressed in the SVD basis of the
    encoding, is compared against ``diag(P(sigma))`` in full.
    """
    operator = QuantumSingularValueTransformation.from_complex_chebyshev_coefficients(
        encoding,
        coefficients,
        synthesis=synthesis,  # type: ignore[arg-type]
    )
    rows, columns = encoding.matrix_shape
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=rows,
        columns=columns,
    )
    left_unitary, singular_values, right_unitary = _normalized_svd(
        encoding.matrix,
        alpha=encoding.alpha,
    )
    expected_values = np.asarray(
        [chebval(float(value), coefficients) for value in singular_values],
        dtype=np.complex128,
    )

    _assert_qsvt_matches_complex_singular_vector_action(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        expected_values=expected_values,
        degree=int(coefficients.shape[0]) - 1,
        atol=atol,
    )


@pytest.mark.parametrize("synthesis", _QSVT_COMPLEX_BACKENDS)
@pytest.mark.parametrize(
    ("family", "qubits"),
    _QSVT_ENCODINGS,
    ids=[f"{family}-{qubits}q" for family, qubits in _QSVT_ENCODINGS],
)
@pytest.mark.parametrize(
    "coefficients",
    list(_QSVT_COMPLEX_TARGETS.values()),
    ids=list(_QSVT_COMPLEX_TARGETS),
)
def test_qsvt_complex_backend_matches_complex_target(
    synthesis: str,
    family: str,
    qubits: int,
    coefficients: np.ndarray,
) -> None:
    _assert_qsvt_matches_complex_target(
        _qsvt_encoding(family, qubits=qubits),
        coefficients,
        synthesis=synthesis,
        atol=_QSVT_COMPLEX_BACKEND_ATOL[synthesis],
    )


@pytest.mark.parametrize("synthesis", _QSVT_COMPLEX_BACKENDS)
@pytest.mark.parametrize(
    ("family", "qubits", "degree", "seed"),
    _QSVT_COMPLEX_RANDOM_CASES,
    ids=[
        f"{family}-{qubits}q-d{degree}-seed{seed}"
        for family, qubits, degree, seed in _QSVT_COMPLEX_RANDOM_CASES
    ],
)
def test_qsvt_complex_backend_matches_random_feasible_target(
    synthesis: str,
    family: str,
    qubits: int,
    degree: int,
    seed: int,
) -> None:
    _assert_qsvt_matches_complex_target(
        _qsvt_encoding(family, qubits=qubits),
        _random_feasible_target(degree, seed=seed),
        synthesis=synthesis,
        atol=_QSVT_COMPLEX_BACKEND_ATOL[synthesis],
    )


def test_qsvt_bfgs_restart_path_is_reproducible() -> None:
    coefficients = _random_feasible_target(7, seed=0)

    first = QuantumSignalProcessing.solve_complex_phases(coefficients)
    second = QuantumSignalProcessing.solve_complex_phases(coefficients)

    np.testing.assert_array_equal(first, second)


def _degenerate_spectrum_encoding(
    spectrum: list[float],
    *,
    seed: int,
) -> UnitaryDilation:
    r"""Return a block encoding whose singular spectrum is degenerate.

    The repeated spectrum is mixed with a random unitary so that the dilation
    stays generic; a bare ``np.diag`` would place the recursive unitary
    synthesis on the Weyl chamber boundary.
    """
    size = len(spectrum)
    rng = np.random.default_rng(seed)
    mixing, _ = np.linalg.qr(
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    )
    return UnitaryDilation(
        np.asarray(mixing @ np.diag(spectrum), dtype=np.complex128),
        alpha=1.0,
        validate=False,
    )


_QSVT_DEGENERATE_SPECTRA: dict[str, list[float]] = {
    "paired": [0.95, 0.95, 0.1, 0.1],
    "triple": [0.9, 0.9, 0.9, 0.1],
}
r"""Degenerate singular spectra; no entry may reach the ``alpha`` boundary."""


@pytest.mark.parametrize("synthesis", _QSVT_COMPLEX_BACKENDS)
@pytest.mark.parametrize(
    "spectrum",
    list(_QSVT_DEGENERATE_SPECTRA.values()),
    ids=list(_QSVT_DEGENERATE_SPECTRA),
)
@pytest.mark.parametrize(
    "coefficients",
    list(_QSVT_COMPLEX_TARGETS.values()),
    ids=list(_QSVT_COMPLEX_TARGETS),
)
def test_qsvt_complex_backend_matches_degenerate_spectrum_target(
    synthesis: str,
    spectrum: list[float],
    coefficients: np.ndarray,
) -> None:
    _assert_qsvt_matches_complex_target(
        _degenerate_spectrum_encoding(spectrum, seed=_QSVT_ENCODING_SEED),
        coefficients,
        synthesis=synthesis,
        atol=_QSVT_COMPLEX_BACKEND_ATOL[synthesis],
    )


_QSVT_COMPLEX_DEGREE_ZERO_TARGETS: dict[str, np.ndarray] = {
    "real-one": np.array([1.0], dtype=np.complex128),
    "imaginary-unit": np.array([1j], dtype=np.complex128),
    "real-minus-one": np.array([-1.0], dtype=np.complex128),
    "quarter-phase": np.array([np.exp(0.25j * math.pi)], dtype=np.complex128),
}
r"""Unit-modulus constants routed through the dedicated degree-0 QSVT channel."""


@pytest.mark.parametrize(
    "coefficients",
    list(_QSVT_COMPLEX_DEGREE_ZERO_TARGETS.values()),
    ids=list(_QSVT_COMPLEX_DEGREE_ZERO_TARGETS),
)
def test_qsvt_complex_degree_zero_matches_constant_target(
    coefficients: np.ndarray,
) -> None:
    r"""The degree-0 channel returns ``arg(c_0)`` and blocks as the constant."""
    encoding = _qsvt_encoding("random", qubits=2)
    operator = QuantumSingularValueTransformation.from_complex_chebyshev_coefficients(
        encoding,
        coefficients,
    )
    expected_phase = math.atan2(coefficients[0].imag, coefficients[0].real)
    assert operator.degree == 0
    np.testing.assert_allclose(
        operator.phases,
        [expected_phase],
        rtol=0.0,
        atol=1e-12,
    )

    rows, columns = encoding.matrix_shape
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=rows,
        columns=columns,
    )
    left_unitary, singular_values, right_unitary = _normalized_svd(
        encoding.matrix,
        alpha=encoding.alpha,
    )
    expected_values = np.asarray(
        [chebval(float(value), coefficients) for value in singular_values],
        dtype=np.complex128,
    )
    _assert_qsvt_matches_complex_singular_vector_action(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        expected_values=expected_values,
        degree=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("synthesis", _QSVT_COMPLEX_BACKENDS)
@pytest.mark.parametrize(
    ("coefficients", "message"),
    list(_QSVT_REJECTED_COMPLEX_TARGETS.values()),
    ids=list(_QSVT_REJECTED_COMPLEX_TARGETS),
)
def test_qsvt_complex_backend_rejects_infeasible_target(
    synthesis: str,
    coefficients: list[complex],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        QuantumSingularValueTransformation.from_complex_chebyshev_coefficients(
            _qsvt_encoding("random", qubits=2),
            coefficients,
            synthesis=synthesis,  # type: ignore[arg-type]
        )


def test_qsvt_rejects_unknown_complex_synthesis_backend() -> None:
    with pytest.raises(ValueError, match="synthesis must be 'bfgs' or 'haah'"):
        QuantumSingularValueTransformation.from_complex_chebyshev_coefficients(
            _qsvt_encoding("random", qubits=2),
            [0.0, 1.0],
            synthesis="dummy",  # type: ignore[arg-type]
        )


_QSVT_REAL_POWER_TARGETS: dict[str, list[float]] = {
    "even-d2": [0.5, 0.0, 0.5],
    "odd-d3": [0.0, 0.5, 0.0, 0.5],
}
_QSVT_REAL_ENCODINGS: list[tuple[str, int]] = [
    (family, qubits) for family in ("random", "spectrum") for qubits in (2, 3)
]
_QSVT_REAL_RANDOM_CASES: list[tuple[str, int, int, int]] = [
    (family, qubits, degree, seed)
    for family in ("random", "spectrum")
    for qubits in (2, 3)
    for degree in (2, 3, 4)
    for seed in range(2)
]

_QSVT_REAL_SYNTHESIS_XFAIL_REASON = (
    "diag(I, U_Phi) circuit synthesis is rejected on this controlled-structured "
    "unitary by two known numerical limitations: cqlib's two-qubit KAK "
    "canonical check (validate_weyl_chamber rejects a = pi/4 with c < 0) and "
    "this repo's unitary_eigendecomposition diagonalisation floor "
    "(failed to diagonalize unitary demux factor). Neither is a QSVT logic error"
)
r"""Reason attached to real-valued cases the synthesis layer currently rejects."""


def _is_known_real_qsvt_synthesis_rejection(error: BaseException) -> bool:
    r"""Return whether ``error`` is a known synthesis-layer rejection.

    Both signatures belong to the controlled ``diag(I, U_{\Phi})`` oracle
    synthesis and are unrelated to the QSVT phase logic under test, so the
    affected cases are reported as runtime ``xfail`` instead of failures.
    """
    if isinstance(error, CompilerTransformError):
        return "Weyl chamber violation" in str(error)
    return isinstance(error, ValueError) and (
        "failed to diagonalize unitary demux factor" in str(error)
    )


def _random_feasible_real_polynomial(degree: int, *, seed: int) -> np.ndarray:
    r"""Return Chebyshev coefficients of a random real QSP-feasible target.

    The coefficients keep the parity of ``degree`` and are rescaled so that
    the polynomial stays strictly inside the unit ball on ``[-1, 1]``, which
    is the boundedness condition required by the real QSP solver.
    """
    rng = np.random.default_rng(seed)
    coefficients = np.zeros(degree + 1, dtype=np.float64)
    indices = list(range(degree % 2, degree + 1, 2))
    coefficients[indices] = rng.uniform(-1.0, 1.0, size=len(indices))
    grid = np.linspace(-1.0, 1.0, max(257, 16 * degree + 1), dtype=np.float64)
    peak = float(np.max(np.abs(chebval(grid, coefficients))))
    return np.asarray(coefficients * (0.9 / peak), dtype=np.float64)


def _assert_real_valued_qsvt_block_matches(
    operator: RealValuedQSVT,
    *,
    degree: int,
    chebyshev_coefficients: list[float],
    atol: float,
) -> None:
    r"""Assert the symmetrised block real part equals ``diag(P(sigma))``."""
    encoding = operator.block_encoding
    rows, columns = encoding.matrix_shape
    projected = _project_standard_block(
        operator.build_circuit(),
        rows=rows,
        columns=columns,
    )
    left_unitary, singular_values, right_unitary = _normalized_svd(
        encoding.matrix,
        alpha=encoding.alpha,
    )
    expected_values = _evaluate_chebyshev_on_singular_values(
        singular_values,
        chebyshev_coefficients=chebyshev_coefficients,
    )

    _assert_qsvt_matches_singular_vector_action(
        projected,
        left_unitary=left_unitary,
        right_unitary=right_unitary,
        expected_values=expected_values,
        degree=degree,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("family", "qubits"),
    _QSVT_REAL_ENCODINGS,
    ids=[f"{family}-{qubits}q" for family, qubits in _QSVT_REAL_ENCODINGS],
)
@pytest.mark.parametrize(
    "coefficients",
    list(_QSVT_REAL_POWER_TARGETS.values()),
    ids=list(_QSVT_REAL_POWER_TARGETS),
)
def test_real_valued_qsvt_matches_power_basis_target(
    family: str,
    qubits: int,
    coefficients: list[float],
) -> None:
    try:
        operator = RealValuedQSVT.from_polynomial_coefficients(
            _qsvt_encoding(family, qubits=qubits),
            coefficients,
        )
        _assert_real_valued_qsvt_block_matches(
            operator,
            degree=len(coefficients) - 1,
            chebyshev_coefficients=list(polynomial_to_chebyshev(coefficients)),
            atol=1e-5,
        )
    except (CompilerTransformError, ValueError) as error:
        if not _is_known_real_qsvt_synthesis_rejection(error):
            raise
        pytest.xfail(_QSVT_REAL_SYNTHESIS_XFAIL_REASON)


@pytest.mark.parametrize(
    ("family", "qubits", "degree", "seed"),
    _QSVT_REAL_RANDOM_CASES,
    ids=[
        f"{family}-{qubits}q-d{degree}-seed{seed}"
        for family, qubits, degree, seed in _QSVT_REAL_RANDOM_CASES
    ],
)
def test_real_valued_qsvt_matches_random_feasible_target(
    family: str,
    qubits: int,
    degree: int,
    seed: int,
) -> None:
    chebyshev_coefficients = _random_feasible_real_polynomial(degree, seed=seed)
    try:
        operator = RealValuedQSVT.from_chebyshev_coefficients(
            _qsvt_encoding(family, qubits=qubits),
            chebyshev_coefficients,
        )
        _assert_real_valued_qsvt_block_matches(
            operator,
            degree=degree,
            chebyshev_coefficients=list(chebyshev_coefficients),
            atol=1e-5,
        )
    except (CompilerTransformError, ValueError) as error:
        if not _is_known_real_qsvt_synthesis_rejection(error):
            raise
        pytest.xfail(_QSVT_REAL_SYNTHESIS_XFAIL_REASON)
