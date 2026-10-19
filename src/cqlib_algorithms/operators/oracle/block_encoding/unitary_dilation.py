r"""Exact one-ancilla block encoding through unitary dilation.

References
----------
Halmos, P. R. (1950). Normal dilations and extensions of operators.
Summa Brasiliensis Mathematicae, 2, 125-134.
"""

from __future__ import annotations

import copy
import math
from numbers import Real
from typing import final

import numpy as np
import numpy.typing as npt

from cqlib import Circuit

from ..unitary_matrix_oracle import UnitaryMatrixOracle
from .block_encoding import BlockEncoding


@final
class UnitaryDilation(BlockEncoding):
    r"""Build a basic exact block encoding from an explicit square matrix.

    This implementation realizes the standard one-ancilla unitary dilation of
    a contraction ``M = A / alpha``:

    .. math::

        U = \begin{bmatrix}
            M & \sqrt{I - MM^\dagger} \\
            \sqrt{I - M^\dagger M} & -M^\dagger
        \end{bmatrix}.

    Under the local qubit layout ``[system..., ancilla]`` the most-significant
    qubit is the ancilla, so the top-left block of ``U`` equals ``A / alpha``.

    Parameters
    ----------
    matrix
        Explicit square matrix to encode. The dimension must be a power of two.
    alpha
        Optional positive subnormalization factor. When omitted, the spectral
        norm of ``matrix`` is used, except that the zero matrix falls back to
        ``1.0`` so that the standard ``(alpha, a, epsilon)`` tuple remains
        well-defined.
    atol
        Non-negative absolute tolerance used for numerical validation.
    validate
        Whether to validate the synthesized dilation circuit through the
        internal unitary-oracle reconstruction check.

    Raises
    ------
    TypeError
        If ``matrix`` is not a NumPy array, or if numeric arguments have
        invalid types.
    ValueError
        If ``matrix`` is not a finite square power-of-two matrix, or if the
        supplied ``alpha`` is smaller than the spectral norm of ``matrix``.
    """

    __slots__ = (
        "_atol",
        "_circuit",
        "_dilation_unitary",
        "_matrix",
        "_normalized_matrix",
        "_validate",
    )

    _matrix: npt.NDArray[np.complex128]
    _normalized_matrix: npt.NDArray[np.complex128]
    _dilation_unitary: npt.NDArray[np.complex128]
    _circuit: Circuit
    _atol: float
    _validate: bool

    def __init__(
        self,
        matrix: npt.NDArray[np.complex128],
        *,
        alpha: float | None = None,
        atol: float = 1e-10,
        validate: bool = True,
    ) -> None:
        normalized_matrix = _validate_input_matrix(matrix)
        tolerance = _validate_tolerance(atol)
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        spectral_norm = _spectral_norm(normalized_matrix)
        effective_alpha = _resolve_alpha(alpha, spectral_norm=spectral_norm, atol=tolerance)
        contraction = np.asarray(
            normalized_matrix / effective_alpha,
            dtype=np.complex128,
        )
        dilation = _build_unitary_dilation(contraction, atol=tolerance)
        oracle = UnitaryMatrixOracle(dilation, atol=tolerance, validate=validate)
        circuit = _align_circuit_global_phase(
            oracle.build_circuit(),
            target_unitary=dilation,
            atol=tolerance,
        )

        super().__init__(
            alpha=effective_alpha,
            ancilla_qubits=1,
            epsilon=0.0,
            matrix_shape=normalized_matrix.shape,
            system_qubits=_num_qubits_from_dimension(normalized_matrix.shape[0]),
        )

        self._matrix = normalized_matrix
        self._normalized_matrix = contraction
        self._dilation_unitary = dilation
        self._circuit = circuit
        self._atol = tolerance
        self._validate = validate

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        r"""Return a defensive copy of the encoded matrix."""

        return np.asarray(self._matrix.copy(), dtype=np.complex128)

    @property
    def normalized_matrix(self) -> npt.NDArray[np.complex128]:
        r"""Return a defensive copy of the contraction ``A / alpha``."""

        return np.asarray(self._normalized_matrix.copy(), dtype=np.complex128)

    @property
    def dilation_unitary(self) -> npt.NDArray[np.complex128]:
        r"""Return a defensive copy of the exact dilation unitary."""

        return np.asarray(self._dilation_unitary.copy(), dtype=np.complex128)

    @property
    def atol(self) -> float:
        r"""Return the numerical validation tolerance."""

        return self._atol

    @property
    def validates_reconstruction(self) -> bool:
        r"""Return whether the internal unitary oracle validated reconstruction."""

        return self._validate

    @property
    def system_register_indices(self) -> list[int]:
        r"""Return the local system-register qubit indices."""

        return list(range(self.system_qubits))

    @property
    def ancilla_register_indices(self) -> list[int]:
        r"""Return the local ancilla-register qubit indices."""

        return list(range(self.system_qubits, self.total_qubits))

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh unitary-dilation circuit."""

        return copy.copy(self._circuit)


def _validate_input_matrix(
    matrix: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    normalized: npt.NDArray[np.complex128] = np.asarray(
        matrix,
        dtype=np.complex128,
    ).copy()
    if normalized.ndim != 2:
        raise ValueError("matrix must be a two-dimensional array")
    rows, columns = normalized.shape
    if rows != columns:
        raise ValueError("matrix must be square")
    if rows == 0:
        raise ValueError("matrix must have positive dimension")
    if rows & (rows - 1):
        raise ValueError("matrix dimension must be a power of two")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("matrix must contain only finite entries")

    return normalized


def _validate_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("atol must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError("atol must be a real finite scalar")
    if scalar < 0.0:
        raise ValueError("atol must be greater than or equal to zero")
    return scalar


def _spectral_norm(matrix: npt.NDArray[np.complex128]) -> float:
    r"""Return the largest singular value (spectral norm) of ``matrix``."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return float(singular_values[0]) if singular_values.size else 0.0


def _resolve_alpha(
    alpha: float | None,
    *,
    spectral_norm: float,
    atol: float,
) -> float:
    r"""Resolve the subnormalization factor, defaulting to the spectral norm."""
    if alpha is None:
        if spectral_norm == 0.0:
            return 1.0
        return spectral_norm

    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real finite scalar")
    scalar = float(alpha)
    if not math.isfinite(scalar):
        raise TypeError("alpha must be a real finite scalar")
    if scalar <= 0.0:
        raise ValueError("alpha must be greater than zero")
    if scalar + atol < spectral_norm:
        raise ValueError("alpha must be at least the spectral norm of matrix")
    return scalar


def _matrix_square_root_psd(
    matrix: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> npt.NDArray[np.complex128]:
    r"""Return the Hermitian positive-semidefinite square root of ``matrix``."""
    hermitian = np.asarray((matrix + matrix.conj().T) / 2.0, dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if float(np.min(eigenvalues)) < -10.0 * max(atol, 1e-12):
        raise ValueError("matrix square root received a non-positive semidefinite input")
    clipped = np.clip(eigenvalues, 0.0, None)
    root = eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.conj().T
    return np.asarray((root + root.conj().T) / 2.0, dtype=np.complex128)


def _build_unitary_dilation(
    contraction: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> npt.NDArray[np.complex128]:
    r"""Construct the exact one-ancilla Halmos dilation of a contraction.

    The returned unitary has the block form

    ``[[M, sqrt(I - M M†)], [sqrt(I - M† M), -M†]]``

    for ``M = contraction``. Under the local layout used by
    :class:`UnitaryDilation`, the ancilla is the most-significant qubit, so the
    top-left block is exactly the projected block encoding ``A / alpha``.
    """

    dimension = contraction.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    upper_right = _matrix_square_root_psd(
        identity - contraction @ contraction.conj().T,
        atol=atol,
    )
    lower_left = _matrix_square_root_psd(
        identity - contraction.conj().T @ contraction,
        atol=atol,
    )
    dilation = np.block(
        [
            [contraction, upper_right],
            [lower_left, -contraction.conj().T],
        ]
    ).astype(np.complex128)

    unitary_check = dilation.conj().T @ dilation
    identity_full = np.eye(2 * dimension, dtype=np.complex128)
    if not np.allclose(unitary_check, identity_full, atol=atol, rtol=0.0):
        raise ValueError("constructed unitary dilation is not unitary within tolerance")

    return dilation


_ALIGN_GLOBAL_PHASE_ATOL_FLOOR: float = 1e-7


def _align_circuit_global_phase(
    circuit: Circuit,
    *,
    target_unitary: npt.NDArray[np.complex128],
    atol: float,
) -> Circuit:
    r"""Shift circuit global phase so the synthesized matrix matches ``target_unitary``.

    ``UnitaryMatrixOracle`` only guarantees equality up to global phase, while
    block encodings care about the literal top-left block. This helper absorbs
    the residual phase into ``Circuit.global_phase`` so that the returned
    circuit matrix matches the target dilation directly, not just modulo phase.
    """

    effective_atol = max(float(atol), _ALIGN_GLOBAL_PHASE_ATOL_FLOOR)

    candidate = copy.copy(circuit)
    candidate_matrix = np.asarray(candidate.to_matrix(), dtype=np.complex128)
    overlap = np.vdot(target_unitary.reshape(-1), candidate_matrix.reshape(-1))
    if abs(overlap) > 0.0:
        phase_shift = float(np.angle(overlap))
        candidate.set_global_phase(candidate.global_phase.evaluate() - phase_shift)

    corrected_matrix = np.asarray(candidate.to_matrix(), dtype=np.complex128)
    if not np.allclose(
        corrected_matrix,
        target_unitary,
        atol=effective_atol,
        rtol=0.0,
    ):
        raise ValueError("synthesized dilation circuit does not match target unitary")
    return candidate


def _num_qubits_from_dimension(dimension: int) -> int:
    r"""Return the qubit count for ``dimension`` basis states."""
    return int(math.log2(dimension))
