"""Oracle wrapper for arbitrary unitary matrices.

References
----------
Krol, A. M., Al-Ars, Z. (2024). Beyond Quantum Shannon: Circuit
Construction for General n-Qubit Gates Based on Block ZXZ-Decomposition.
arXiv:2403.13692.
"""

from __future__ import annotations

import copy
from typing import final

import numpy as np
import numpy.typing as npt

from cqlib import Circuit

from ._synthesis import synthesize_unitary_matrix_circuit
from ._synthesis._common import assert_square_unitary, num_qubits_from_matrix
from ._validation import (
    allclose_up_to_global_phase,
    reconstruction_error_norm,
    validate_atol,
)


@final
class UnitaryMatrixOracle:
    """Snapshot an arbitrary unitary matrix as an oracle circuit.

    The input matrix is validated, synthesized once through the internal
    Block-ZXZ pipeline, and stored as a normalized circuit snapshot. The
    recursive synthesis terminates at one- and two-qubit leaves using the
    native ``cqlib`` numeric unitary decomposers.

    Parameters
    ----------
    unitary
        Target unitary matrix with shape ``(2**n, 2**n)``.
    atol
        Non-negative absolute tolerance used for validation and the optional
        reconstruction check.
    validate
        Whether to verify the synthesized circuit matrix against ``unitary``
        up to global phase before storing the snapshot.

    Raises
    ------
    TypeError
        If ``unitary`` is not a NumPy array, or if ``atol`` / ``validate`` have
        invalid types.
    ValueError
        If ``unitary`` is not a finite square power-of-two unitary, or if the
        validated reconstruction exceeds ``atol``.
    """

    __slots__ = ("_atol", "_circuit", "_unitary", "_validate")

    _unitary: npt.NDArray[np.complex128]
    _atol: float
    _validate: bool
    _circuit: Circuit

    def __init__(
        self,
        unitary: npt.NDArray[np.complex128],
        *,
        atol: float = 1e-10,
        validate: bool = True,
    ) -> None:
        if not isinstance(unitary, np.ndarray):
            raise TypeError("unitary must be a numpy.ndarray")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        normalized = np.asarray(unitary, dtype=np.complex128).copy()
        tolerance = validate_atol(atol)
        assert_square_unitary(normalized, atol=tolerance)

        circuit = synthesize_unitary_matrix_circuit(normalized, atol=tolerance)
        if validate:
            reconstructed = np.asarray(circuit.to_matrix(), dtype=np.complex128)
            if not allclose_up_to_global_phase(
                normalized,
                reconstructed,
                atol=tolerance,
            ):
                error_norm = reconstruction_error_norm(normalized, reconstructed)
                raise ValueError(
                    "synthesized circuit does not reconstruct the input unitary "
                    f"within atol={tolerance:g} (error={error_norm:g})"
                )

        self._unitary = normalized
        self._atol = tolerance
        self._validate = validate
        self._circuit = circuit

    @property
    def unitary(self) -> npt.NDArray[np.complex128]:
        """Return a defensive copy of the target unitary matrix."""

        return np.asarray(self._unitary.copy(), dtype=np.complex128)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Return the unitary matrix shape."""

        return (self._unitary.shape[0], self._unitary.shape[1])

    @property
    def num_qubits(self) -> int:
        """Return the qubit width of the oracle."""

        return num_qubits_from_matrix(self._unitary)

    @property
    def atol(self) -> float:
        """Return the configured validation tolerance."""

        return self._atol

    @property
    def validates_reconstruction(self) -> bool:
        """Return whether post-synthesis reconstruction was enforced."""

        return self._validate

    @property
    def dimension(self) -> int:
        """Return the Hilbert-space dimension of the oracle."""

        return int(self._unitary.shape[0])

    def build_circuit(self) -> Circuit:
        """Return an independent oracle circuit snapshot."""

        return copy.copy(self._circuit)

    def build_adjoint_circuit(self) -> Circuit:
        """Return an independent adjoint oracle circuit snapshot."""

        circuit = copy.copy(self._circuit)
        return circuit.inverse()
