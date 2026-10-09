"""Tests for standard QSVT projector phases and alternating sequences."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval

from cqlib import Circuit, Qubit
from cqlib_algorithms.operators.oracle import (
    BlockEncoding,
    ProjectorControlledPhase,
    QuantumSignalProcessing,
    QuantumSingularValueTransformation,
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


def _assert_qsvt_matches_singular_vector_action(
    projected: np.ndarray,
    *,
    left_unitary: np.ndarray,
    right_unitary: np.ndarray,
    expected_values: np.ndarray,
    degree: int,
    atol: float,
) -> None:
    if degree % 2 == 0:
        singular_basis_matrix = right_unitary.conj().T @ projected @ right_unitary
    else:
        singular_basis_matrix = left_unitary.conj().T @ projected @ right_unitary

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
        QuantumSingularValueTransformation.convert_qsp_phases(qsp_phases),
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
    unitary = np.asarray(block_encoding.build_circuit().to_matrix(), dtype=np.complex128)
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
    with pytest.raises(ValueError, match="greater than or equal to zero"):
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


def test_qsvt_degree_one_scalar_matches_qsp_response() -> None:
    scalar = 0.4
    coefficients = [0.0, 1.0]
    encoding = UnitaryDilation(np.array([[scalar]], dtype=np.complex128), validate=False)
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    qsvt = QuantumSingularValueTransformation(
        encoding,
        QuantumSingularValueTransformation.convert_qsp_phases(qsp_phases),
    )
    qsp = QuantumSignalProcessing(qsp_phases, signal_value=scalar / encoding.alpha)

    projected = _project_standard_block(
        qsvt.build_circuit(),
        rows=1,
        columns=1,
    )
    qsp_top_left = np.asarray(qsp.build_circuit().to_matrix(), dtype=np.complex128)[0, 0]

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
        QuantumSingularValueTransformation.convert_qsp_phases(qsp_phases),
    )
    qsp = QuantumSignalProcessing(qsp_phases, signal_value=scalar / alpha)

    projected = _project_standard_block(
        qsvt.build_circuit(),
        rows=1,
        columns=1,
    )
    qsp_top_left = np.asarray(qsp.build_circuit().to_matrix(), dtype=np.complex128)[0, 0]

    np.testing.assert_allclose(
        projected[0, 0],
        qsp_top_left,
        rtol=0.0,
        atol=1e-12,
    )


def test_qsvt_degree_two_scalar_matches_qsp_response() -> None:
    scalar = 0.4
    coefficients = [0.0, 0.0, 1.0]
    encoding = UnitaryDilation(np.array([[scalar]], dtype=np.complex128), validate=False)
    qsp_phases = QuantumSignalProcessing.solve_phases(
        coefficients,
        tolerance=1e-12,
        max_iterations=200,
        gradient_tolerance=1e-9,
    )
    qsvt = QuantumSingularValueTransformation(
        encoding,
        QuantumSingularValueTransformation.convert_qsp_phases(qsp_phases),
    )
    qsp = QuantumSignalProcessing(qsp_phases, signal_value=scalar / encoding.alpha)

    projected = _project_standard_block(
        qsvt.build_circuit(),
        rows=1,
        columns=1,
    )
    qsp_top_left = np.asarray(qsp.build_circuit().to_matrix(), dtype=np.complex128)[0, 0]

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


def test_qsvt_exact_dilation_matrix_matches_circuit_composition_for_high_degree_case() -> None:
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
    qsvt_phases = QuantumSingularValueTransformation.convert_qsp_phases(qsp_phases)
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
    converted = QuantumSingularValueTransformation.convert_qsp_phases(qsp_phases)

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


def test_qsvt_check_block_encoding_requires_explicit_matrix_for_generic_encoding() -> None:
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
        QuantumSingularValueTransformation.convert_qsp_phases(expected_phases),
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


def test_qsvt_check_block_encoding_rejects_invalid_expected_matrix_and_tolerance() -> None:
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
