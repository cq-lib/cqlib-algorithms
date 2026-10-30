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

"""Tests for arbitrary unitary-matrix oracle synthesis."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from cqlib import Circuit, Qubit
from cqlib.compile import CompilerTransformError
from cqlib.compile.transform.decompose.unitary import (
    synthesize_numeric_2q_unitary,
)
from cqlib_algorithms.operators.oracle import UnitaryDilation, UnitaryMatrixOracle
from cqlib_algorithms.operators.composite.quantum_singular_value_transformation.real_qsvt import (
    RealValuedQSVT,
)
from cqlib_algorithms.operators.oracle.unitary_matrix_oracle._synthesis import (
    _demultiplex as demultiplex_module,
)
from cqlib_algorithms.operators.oracle.unitary_matrix_oracle._synthesis._common import (
    is_hermitian_for_demux,
    safe_project_subunitary,
    unitary_eigendecomposition,
)


def _align_global_phase(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    overlap = np.vdot(reference.reshape(-1), candidate.reshape(-1))
    if abs(overlap) == 0.0:
        return np.asarray(candidate.copy(), dtype=np.complex128)
    return np.asarray(
        candidate * np.exp(-1j * np.angle(overlap)),
        dtype=np.complex128,
    )


def _assert_allclose_up_to_global_phase(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float = 1e-9,
) -> None:
    aligned = _align_global_phase(reference, candidate)
    np.testing.assert_allclose(reference, aligned, rtol=0.0, atol=atol)


def _haar_random_unitary(num_qubits: int, *, seed: int) -> np.ndarray:
    dim = 2**num_qubits
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q_factor, r_factor = np.linalg.qr(gaussian)
    phases = np.diag(r_factor)
    normalized = np.divide(
        phases,
        np.abs(phases),
        out=np.ones_like(phases, dtype=np.complex128),
        where=np.abs(phases) > 0.0,
    )
    return np.asarray(q_factor @ np.diag(np.conj(normalized)), dtype=np.complex128)


def _diagonal_phase_unitary(num_qubits: int) -> np.ndarray:
    dimension = 2**num_qubits
    phases = np.linspace(0.0, 0.7, dimension, dtype=np.float64)
    return np.asarray(np.diag(np.exp(1j * phases)), dtype=np.complex128)


def test_one_qubit_unitary_matrix_oracle_reconstructs_up_to_global_phase():
    matrix = np.exp(0.37j) * np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)

    circuit = UnitaryMatrixOracle(matrix).build_circuit()

    _assert_allclose_up_to_global_phase(matrix, circuit.to_matrix(), atol=1e-10)


def test_two_qubit_unitary_matrix_oracle_reconstructs_swap_exactly():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    circuit = UnitaryMatrixOracle(matrix).build_circuit()

    np.testing.assert_allclose(matrix, circuit.to_matrix(), rtol=0.0, atol=1e-10)


def test_three_qubit_random_unitary_matrix_oracle_reconstructs_up_to_global_phase():
    matrix = _haar_random_unitary(3, seed=7)

    circuit = UnitaryMatrixOracle(matrix).build_circuit()

    _assert_allclose_up_to_global_phase(matrix, circuit.to_matrix(), atol=1e-8)


_LARGE_RANDOM_UNITARY_CASES = [
    pytest.param(num_qubits, seed, id=f"{num_qubits}q-seed{seed}")
    for num_qubits in (3, 4, 5)
    for seed in range(5)
]


def _random_square_matrix(dimension: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension)),
        dtype=np.complex128,
    )


def _scaled_unitary_dilation_matrix(
    dimension: int,
    *,
    seed: int,
) -> np.ndarray:
    matrix = _random_square_matrix(dimension, seed=seed)
    spectral_norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    alpha = 2.0 * spectral_norm
    component = UnitaryDilation(matrix, alpha=alpha, validate=False)
    return np.asarray(component.dilation_unitary, dtype=np.complex128)


_RANDOM_UNITARY_DILATION_CASES = [
    pytest.param(num_qubits, seed, id=f"{num_qubits}q-seed{seed}")
    for num_qubits in (3, 4, 5)
    for seed in range(5)
]

_REPRO_UNITARY_DILATION_CASES = [
    pytest.param(5, 707, id="5q-seed707-qsvt-odd"),
    pytest.param(5, 808, id="5q-seed808-qsvt-even"),
]


def _error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate - reference, dtype=np.complex128)
    if difference.size == 0:
        return {"max_abs": 0.0, "frobenius": 0.0, "spectral": 0.0}
    max_abs = float(np.max(np.abs(difference)))
    frobenius = float(np.linalg.norm(difference, ord="fro"))
    singular_values = np.linalg.svd(difference, compute_uv=False)
    spectral = float(singular_values[0]) if singular_values.size else 0.0
    return {
        "max_abs": max_abs,
        "frobenius": frobenius,
        "spectral": spectral,
    }


def _print_error_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: max_abs={metrics['max_abs']:.6e}, "
        f"frobenius={metrics['frobenius']:.6e}, "
        f"spectral={metrics['spectral']:.6e}"
    )


def _make_diagnostic_demultiplex_pair(original: Any) -> Any:
    def _diagnostic_demultiplex_pair(
        first_unitary: np.ndarray,
        second_unitary: np.ndarray,
        *,
        atol: float,
    ):
        try:
            return original(first_unitary, second_unitary, atol=atol)
        except ValueError as exc:
            message = str(exc)
            if "reconstruction of the first unitary failed" in message:
                first_projected = safe_project_subunitary(
                    first_unitary,
                    atol=atol,
                    label="diagnostic demultiplex first input",
                )
                second_projected = safe_project_subunitary(
                    second_unitary,
                    atol=atol,
                    label="diagnostic demultiplex second input",
                )
                product = np.asarray(
                    first_projected @ second_projected.conj().T,
                    dtype=np.complex128,
                )
                product = safe_project_subunitary(
                    product,
                    atol=atol,
                    label="diagnostic demultiplex eig product",
                )
                left_factor, eigenvalues = unitary_eigendecomposition(product, atol=atol)
                diagonal_factor = (
                    np.diag(np.sign(eigenvalues))
                    if eigenvalues.dtype.kind in "biu"
                    else np.zeros_like(eigenvalues)
                )
                _ = diagonal_factor
                print(
                    "demux first reconstruction: "
                    f"shape={first_projected.shape}, atol={atol:.3e}"
                )
            elif "reconstruction of the second unitary failed" in message:
                print(
                    "demux second reconstruction: "
                    f"shape={np.asarray(first_unitary).shape}, atol={atol:.3e}"
                )
            elif "failed to diagonalize unitary demux factor" in message:
                first_projected = safe_project_subunitary(
                    first_unitary,
                    atol=atol,
                    label="diagnostic demultiplex first input",
                )
                second_projected = safe_project_subunitary(
                    second_unitary,
                    atol=atol,
                    label="diagnostic demultiplex second input",
                )
                product = np.asarray(
                    first_projected @ second_projected.conj().T,
                    dtype=np.complex128,
                )
                product = safe_project_subunitary(
                    product,
                    atol=atol,
                    label="diagnostic demultiplex eig product",
                )
                identity = np.eye(product.shape[0], dtype=np.complex128)
                unitarity = _error_metrics(identity, product.conj().T @ product)
                _print_error_metrics("demux product unitarity residual", unitarity)
                hermitian_residual = _error_metrics(
                    product.conj().T,
                    product,
                )
                _print_error_metrics(
                    "demux product Hermitian residual",
                    hermitian_residual,
                )
                if is_hermitian_for_demux(product):
                    print("demux diagonalization branch: eigh")
                    eigenvalues, eigenvectors = np.linalg.eigh(product)
                    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
                    left_factor = np.asarray(eigenvectors, dtype=np.complex128)
                else:
                    print("demux diagonalization branch: eig")
                    eigenvalues, eigenvectors = np.linalg.eig(product)
                    order = np.argsort(np.angle(eigenvalues))
                    eigenvalues = eigenvalues[order]
                    left_factor = np.asarray(eigenvectors[:, order], dtype=np.complex128)
                    left_factor, _ = np.linalg.qr(left_factor)
                diagonal_matrix = left_factor.conj().T @ product @ left_factor
                diagonal_matrix_abs = np.abs(diagonal_matrix)
                off_diagonal_mask = ~np.eye(
                    diagonal_matrix_abs.shape[0],
                    dtype=bool,
                )
                if off_diagonal_mask.any():
                    off_diag_value = np.asarray(
                        diagonal_matrix * off_diagonal_mask,
                        dtype=np.complex128,
                    )
                    off_diag_metrics = _error_metrics(
                        np.zeros_like(off_diag_value),
                        off_diag_value,
                    )
                    _print_error_metrics(
                        "demux diagonalization off-diagonal residual",
                        off_diag_metrics,
                    )
                eigenvalue_modulus_error = float(
                    np.max(np.abs(np.abs(eigenvalues) - 1.0))
                )
                print(
                    "demux eigenvalue modulus max error="
                    f"{eigenvalue_modulus_error:.6e}"
                )
            raise

    return _diagnostic_demultiplex_pair


def _build_oracle_with_diagnostics(
    matrix: np.ndarray,
    *,
    atol: float,
) -> Circuit:
    original_demux_pair = demultiplex_module.demultiplex_pair
    patched_pair = _make_diagnostic_demultiplex_pair(original_demux_pair)
    try:
        demultiplex_module.demultiplex_pair = patched_pair
        oracle = UnitaryMatrixOracle(matrix, atol=atol, validate=False)
        return oracle.build_circuit()
    except ValueError as exc:
        reason = str(exc)
        detail_lines = []
        if reason:
            detail_lines.append(reason)
        if detail_lines:
            print("oracle failure context: " + " | ".join(detail_lines))
        pytest.fail(reason or "oracle synthesis failed", pytrace=False)
    finally:
        demultiplex_module.demultiplex_pair = original_demux_pair


def _assert_dilation_oracle_reconstruction_with_diagnostics(
    matrix: np.ndarray,
    *,
    atol: float,
) -> None:
    circuit = _build_oracle_with_diagnostics(matrix, atol=atol)
    circuit_matrix = np.asarray(circuit.to_matrix(), dtype=np.complex128)
    aligned = _align_global_phase(matrix, circuit_matrix)
    oracle_metrics = _error_metrics(matrix, aligned)
    if oracle_metrics["max_abs"] > atol:
        _print_error_metrics(
            "oracle reconstruction up to global phase mismatch",
            oracle_metrics,
        )
        best_phase = float(np.angle(np.vdot(matrix.reshape(-1), circuit_matrix.reshape(-1))))
        literal_aligned = np.asarray(
            circuit_matrix * np.exp(-1j * best_phase),
            dtype=np.complex128,
        )
        literal_metrics = _error_metrics(matrix, literal_aligned)
        _print_error_metrics(
            "dilation literal reconstruction mismatch",
            literal_metrics,
        )
        pytest.fail(
            "oracle reconstruction exceeds tolerance "
            f"(max_abs={oracle_metrics['max_abs']:.6e} > atol={atol:.3e})",
            pytrace=False,
        )


@pytest.mark.parametrize(("num_qubits", "seed"), _LARGE_RANDOM_UNITARY_CASES)
def test_larger_random_unitaries_reconstruct_up_to_global_phase(
    num_qubits: int,
    seed: int,
):
    matrix = _haar_random_unitary(num_qubits, seed=seed)

    circuit = UnitaryMatrixOracle(matrix, atol=1e-8).build_circuit()

    _assert_allclose_up_to_global_phase(matrix, circuit.to_matrix(), atol=1e-8)


@pytest.mark.parametrize("num_qubits", [4, 5, 6])
def test_diagonal_unitary_reconstructs_up_to_global_phase(num_qubits: int):
    matrix = _diagonal_phase_unitary(num_qubits)

    circuit = UnitaryMatrixOracle(matrix, atol=1e-8).build_circuit()

    _assert_allclose_up_to_global_phase(matrix, circuit.to_matrix(), atol=1e-8)


def test_block_diagonal_identity_controlled_unitary_reconstructs_up_to_global_phase():
    """``diag(I, U)`` arises when building a control-on-|1> gate from an
    arbitrary core unitary ``U``.  The Block-ZXZ recursion must remain
    numerically stable on this exactly block-diagonal manifold."""

    core = _haar_random_unitary(2, seed=3)
    dimension = core.shape[0]
    matrix = np.block(
        [
            [np.eye(dimension, dtype=np.complex128), np.zeros((dimension, dimension), dtype=np.complex128)],
            [np.zeros((dimension, dimension), dtype=np.complex128), core],
        ]
    )

    circuit = UnitaryMatrixOracle(matrix, atol=1e-8).build_circuit()

    _assert_allclose_up_to_global_phase(matrix, circuit.to_matrix(), atol=1e-8)


@pytest.mark.parametrize(
    "matrix",
    [
        #np.array([[0.3, 0.2j], [0.0, 0.7]], dtype=np.complex128),
        np.array([[0.3, 0.1j], [0.2j, 0.7]], dtype=np.complex128),
        np.array([[0.2, 0.3], [0.0, 0.5]], dtype=np.complex128),
        np.array([[0.1, 0.4j], [0.0, 0.9]], dtype=np.complex128),
        np.array([[0.6, 0.1 + 0.1j], [0.0, 0.4]], dtype=np.complex128),
        np.array([[0.4, 0.2 - 0.3j], [0.0, 0.6]], dtype=np.complex128),
        np.array([[0.15, 0.5j], [0.0, 0.8]], dtype=np.complex128),
        np.array([[0.7, 0.05 + 0.05j], [0.0, 0.3]], dtype=np.complex128),
        np.array(
            [
                [0.3, 0.1 + 0.2j, 0.05, 0.0],
                [0.0, 0.6, 0.1j, -0.1],
                [0.0, 0.0, 0.5, 0.3],
                [0.0, 0.0, 0.0, 0.4],
            ],
            dtype=np.complex128,
        ),
        np.array(
            [
                [0.25, 0.1 + 0.1j, 0.0, 0.05j],
                [0.0, 0.55, 0.05 - 0.1j, 0.1],
                [0.0, 0.0, 0.45, 0.2],
                [0.0, 0.0, 0.0, 0.35],
            ],
            dtype=np.complex128,
        ),
    ],
    ids=[
        "2x2_lower_nonzero",
        "2x2_real_offdiag",
        "2x2_imag_offdiag_high_spectrum",
        "2x2_complex_offdiag_small",
        "2x2_complex_offdiag_mid",
        "2x2_imag_offdiag",
        "2x2_complex_offdiag_tiny",
        "4x4_upper_triangular_1",
        "4x4_upper_triangular_2",
    ],
)
def test_block_diagonal_qsvt_core_unitary_reconstructs_up_to_global_phase(matrix):
    """``diag(I, U_Phi)`` where ``U_Phi`` is the QSVT core circuit for a
    non-diagonal matrix.  This specific structure triggered a KAK Weyl
    chamber violation in the 2-qubit base case of the Block-ZXZ recursion."""

    encoding = UnitaryDilation(matrix, alpha=1.0, validate=False)
    operator = RealValuedQSVT.from_polynomial_coefficients(
        encoding,
        (0.5, 0.0, 0.5),
    )

    core = np.asarray(
        operator._positive_core.build_circuit().to_matrix(),
        dtype=np.complex128,
    )
    dimension = core.shape[0]
    controlled = np.block(
        [
            [np.eye(dimension, dtype=np.complex128), np.zeros((dimension, dimension), dtype=np.complex128)],
            [np.zeros((dimension, dimension), dtype=np.complex128), core],
        ]
    )

    circuit = UnitaryMatrixOracle(controlled, atol=1e-7).build_circuit()

    _assert_allclose_up_to_global_phase(controlled, circuit.to_matrix(), atol=1e-7)


_KAK_WEYL_BOUNDARY_XFAIL_REASON = (
    "the matrix is unitary but cqlib's two-qubit KAK canonical check "
    "(validate_weyl_chamber) rejects the Weyl chamber boundary a = pi/4 with "
    "c < 0, so the synthesis layer cannot be exercised on this case yet"
)
"""Reason attached to the KAK boundary case the compiler currently rejects."""


def test_two_qubit_kak_weyl_boundary_a_pi_4_c_negative():
    """A 4x4 unitary whose canonical KAK coordinates land on the Weyl
    chamber boundary ``a = pi/4`` with a negative ``c``.  cqlib's KAK
    implementation requires ``c >= 0`` when ``a = pi/4``; this matrix
    reproduces the failure reported by the QSVT ``diag(I, U_Phi)`` case."""

    matrix = np.array(
        [
            [
                -1.4755155196621783e-01 + 8.5353452270019212e-02j,
                -1.5112015359785388e-02 - 2.6124324914469213e-02j,
                -4.3021142204224816e-16 + 6.6019804395935772e-02j,
                -9.8268706095328273e-01 + 2.7755575615628914e-16j,
            ],
            [
                -6.2122904195909606e-03 - 1.0739262071974568e-02j,
                -3.5893385033025271e-01 + 2.0763077618642661e-01j,
                9.0707578731568661e-01 - 5.4123372450476381e-16j,
                -6.2450045135165055e-16 - 7.1523028674972039e-02j,
            ],
            [
                -9.7069336463786215e-01 + 2.7755575615628914e-17j,
                2.6020852139652106e-18 - 1.5594957493615416e-01j,
                3.5697051715777181e-02 - 6.1709927849413246e-02j,
                1.4575068516388590e-01 + 8.4311713321094467e-02j,
            ],
            [
                7.1123662515049091e-17 - 1.6894909068671718e-01j,
                -8.9600492665168041e-01 - 3.6082248300317588e-16j,
                -3.5455306241796475e-01 - 2.0509664240745978e-01j,
                -1.4674445936026376e-02 + 2.5367893324985950e-02j,
            ],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        np.eye(matrix.shape[0], dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )

    try:
        synthesis = synthesize_numeric_2q_unitary(matrix, Qubit(0), Qubit(1))
    except CompilerTransformError as error:
        if "Weyl chamber violation" not in str(error):
            raise
        pytest.xfail(_KAK_WEYL_BOUNDARY_XFAIL_REASON)

    circuit = Circuit.from_operations([Qubit(0), Qubit(1)], synthesis.operations)
    circuit.set_global_phase(synthesis.global_phase)

    _assert_allclose_up_to_global_phase(matrix, circuit.to_matrix(), atol=1e-8)


@pytest.mark.parametrize(("num_qubits", "seed"), _RANDOM_UNITARY_DILATION_CASES)
def test_random_unitary_dilation_reconstructs_up_to_global_phase(
    num_qubits: int,
    seed: int,
):
    dimension = 2**num_qubits
    matrix = _scaled_unitary_dilation_matrix(dimension, seed=seed)

    _assert_dilation_oracle_reconstruction_with_diagnostics(matrix, atol=1e-8)


@pytest.mark.parametrize(("num_qubits", "seed"), _REPRO_UNITARY_DILATION_CASES)
def test_unitary_dilation_from_qsvt_bad_case_reconstructs_up_to_global_phase(
    num_qubits: int,
    seed: int,
):
    dimension = 2**num_qubits
    matrix = _scaled_unitary_dilation_matrix(dimension, seed=seed)

    _assert_dilation_oracle_reconstruction_with_diagnostics(matrix, atol=1e-8)


def test_build_returns_fresh_circuits_and_adjoint_matches_inverse():
    matrix = _haar_random_unitary(2, seed=11)
    oracle = UnitaryMatrixOracle(matrix)

    first = oracle.build_circuit()
    second = oracle.build_circuit()
    adjoint = oracle.build_adjoint_circuit()

    assert first is not second
    assert first == second
    first.h(0)
    assert first != second
    _assert_allclose_up_to_global_phase(
        matrix.conj().T,
        adjoint.to_matrix(),
        atol=1e-10,
    )


def test_oracle_composes_positionally_onto_non_contiguous_qubits():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.complex128,
    )
    destination = Circuit([7, 2])

    destination.compose(
        UnitaryMatrixOracle(matrix).build_circuit(),
        [Qubit(7), Qubit(2)],
    )

    _assert_allclose_up_to_global_phase(
        matrix,
        destination.to_matrix([7, 2]),
        atol=1e-10,
    )


def test_configuration_is_keyword_only_and_properties_are_exposed():
    matrix = np.eye(2, dtype=np.complex128)
    signature = inspect.signature(UnitaryMatrixOracle)
    oracle = UnitaryMatrixOracle(matrix, atol=1e-9, validate=False)

    assert signature.parameters["atol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert oracle.num_qubits == 1
    assert oracle.dimension == 2
    assert oracle.matrix_shape == (2, 2)
    assert oracle.atol == pytest.approx(1e-9)
    assert oracle.validates_reconstruction is False
    np.testing.assert_allclose(oracle.unitary, matrix, rtol=0.0, atol=0.0)

    copy_of_matrix = oracle.unitary
    copy_of_matrix[0, 0] = 0.0
    np.testing.assert_allclose(oracle.unitary, matrix, rtol=0.0, atol=0.0)

    with pytest.raises(TypeError):
        UnitaryMatrixOracle(matrix, 1e-9)  # type: ignore[misc]


def test_unitary_matrix_oracle_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="numpy.ndarray"):
        UnitaryMatrixOracle([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real finite scalar"):
        UnitaryMatrixOracle(np.eye(2, dtype=np.complex128), atol=True)
    with pytest.raises(TypeError, match="validate must be a bool"):
        UnitaryMatrixOracle(np.eye(2, dtype=np.complex128), validate=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="power of two"):
        UnitaryMatrixOracle(np.eye(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="unitary within"):
        UnitaryMatrixOracle(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128))
