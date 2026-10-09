"""Standard quantum singular value transformation circuit construction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
import numpy.typing as npt

from cqlib import Circuit

from ..block_encoding import BlockEncoding
from .projector_controlled_phase import ProjectorControlledPhase
from .qsp import QuantumSignalProcessing, _normalize_phases


class QuantumSingularValueTransformation:
    """Build the standard QSVT alternating sequence.

    This implementation currently supports only standard block encodings with

    .. math::

        \\Pi = \\tilde{\\Pi} = |0\\rangle\\langle 0|^{\\otimes a} \\otimes I.

    The alternating sequence follows Definition 15 in Ref. [1]. The phases are
    computed by the real-QSP phase synthesis solver implemented in
    ``QuantumSignalProcessing.solve_phases``, following the phase-factor
    evaluation framework of Dong et al. [2]. The returned full symmetric QSP
    phase sequence is then converted to the phase sequence used by this QSVT
    construction via the ``R``-form phase equivalence derived in the proof of
    Corollary 8 in Ref. [1].

    Parameters
    ----------
    block_encoding
        Standard block encoding used as the QSVT signal operator.
    phases
        QSVT phase sequence used by the alternating construction.

    Raises
    ------
    TypeError
        If ``block_encoding`` or ``phases`` has an invalid type.
    ValueError
        If ``block_encoding`` is not standard.

    References
    ----------
    .. [1] A. Gilyen, Y. Su, G. H. Low, and N. Wiebe,
       "Quantum singular value transformation and beyond:
       exponential improvements for quantum matrix arithmetics,"
       Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of
       Computing, pp. 193-204, 2019. arXiv:1806.01838.
    .. [2] Y. Dong, X. Meng, K. B. Whaley, and L. Lin,
       "Efficient phase-factor evaluation in quantum signal processing,"
       Physical Review A 103, 042419, 2021. arXiv:2002.11649.
    """

    __slots__ = (
        "_block_encoding",
        "_left_projector",
        "_phases",
        "_right_projector",
    )

    _block_encoding: BlockEncoding
    _phases: tuple[float, ...]
    _right_projector: ProjectorControlledPhase
    _left_projector: ProjectorControlledPhase

    def __init__(
        self,
        block_encoding: BlockEncoding,
        phases: Sequence[float] | np.ndarray,
    ) -> None:
        if not isinstance(block_encoding, BlockEncoding):
            raise TypeError("block_encoding must be a BlockEncoding")
        if not block_encoding.is_standard:
            raise ValueError("QSVT currently supports only standard block encodings")

        self._block_encoding = block_encoding
        self._phases = _normalize_qsvt_phases(phases)
        self._right_projector = ProjectorControlledPhase(
            block_encoding.system_qubits,
            block_encoding.ancilla_qubits,
            phase=0.0,
        )
        self._left_projector = ProjectorControlledPhase(
            block_encoding.system_qubits,
            block_encoding.ancilla_qubits,
            phase=0.0,
        )

    @property
    def block_encoding(self) -> BlockEncoding:
        """Return the underlying standard block encoding."""

        return self._block_encoding

    @property
    def phases(self) -> tuple[float, ...]:
        """Return the immutable QSVT phase sequence."""

        return self._phases

    @property
    def right_projector(self) -> ProjectorControlledPhase:
        """Return the standard right-projector phase template."""

        return self._right_projector

    @property
    def left_projector(self) -> ProjectorControlledPhase:
        """Return the standard left-projector phase template."""

        return self._left_projector

    @property
    def num_qubits(self) -> int:
        """Return the width of the local QSVT circuit."""

        return self.block_encoding.total_qubits

    def build_circuit(self) -> Circuit:
        """Build and return a fresh local-numbering QSVT circuit."""

        circuit = Circuit(self.num_qubits)
        phase_count = len(self.phases)
        for reverse_index, phase in enumerate(reversed(self.phases), start=1):
            phase_index = phase_count - reverse_index + 1
            if phase_index % 2 == phase_count % 2:
                circuit.compose(self.block_encoding.build_circuit())
                circuit.compose(self.left_projector.with_phase(phase).build_circuit())
            else:
                circuit.compose(self.block_encoding.build_adjoint_circuit())
                circuit.compose(self.right_projector.with_phase(phase).build_circuit())
        return circuit

    @staticmethod
    def convert_qsp_phases(
        phases: Sequence[float] | np.ndarray,
    ) -> tuple[float, ...]:
        """Convert full symmetric QSP phases to QSVT phases.

        Parameters
        ----------
        phases
            Full symmetric QSP phase sequence ``(phi'_0, ..., phi'_d)``.

        Returns
        -------
        tuple[float, ...]
            Converted QSVT phase sequence ``(phi_1, ..., phi_d)``.
        """

        normalized = _normalize_phases(phases)
        degree = len(normalized) - 1
        if degree == 0:
            return ()

        converted = [
            normalized[0] + normalized[-1] + 0.5 * (degree - 1) * math.pi,
        ]
        converted.extend(phase - 0.5 * math.pi for phase in normalized[1:-1])
        return tuple(converted)

    def check_block_encoding(
        self,
        *,
        expected_matrix: npt.NDArray[np.complex128] | None = None,
        atol: float = 1e-10,
    ) -> float:
        """Validate the standard projected identity ``A = PiTilde U Pi``.

        Parameters
        ----------
        expected_matrix
            Explicit target matrix ``A``. When omitted, the method uses the
            ``matrix`` attribute of the current block encoding if available.
        atol
            Absolute tolerance for the projected-block comparison.

        Returns
        -------
        float
            Frobenius norm of the projected-block reconstruction error.

        Raises
        ------
        TypeError
            If the expected matrix or tolerance has an invalid type.
        ValueError
            If no explicit target matrix is available, if the target matrix has
            an invalid shape, or if the projected identity is not satisfied.
        """

        tolerance = _coerce_tolerance(atol)
        target = _resolve_expected_matrix(
            block_encoding=self.block_encoding,
            expected_matrix=expected_matrix,
        )
        projected = _project_standard_block(
            self.block_encoding.build_circuit(),
            rows=self.block_encoding.matrix_shape[0],
            columns=self.block_encoding.matrix_shape[1],
        )
        normalized_target = np.asarray(
            target / self.block_encoding.alpha,
            dtype=np.complex128,
        )
        error_norm = float(np.linalg.norm(projected - normalized_target))
        if not np.allclose(projected, normalized_target, atol=tolerance, rtol=0.0):
            raise ValueError(
                "block encoding does not satisfy A = PiTilde U Pi within "
                f"atol={tolerance:g} (error={error_norm:g})"
            )
        return error_norm

    @classmethod
    def from_chebyshev_coefficients(
        cls,
        block_encoding: BlockEncoding,
        coefficients: Sequence[float] | np.ndarray,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSingularValueTransformation:
        """Solve QSP phases, convert and bind them to a block encoding."""

        return cls(
            block_encoding,
            cls.convert_qsp_phases(
                QuantumSignalProcessing.solve_phases(
                    coefficients,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    gradient_tolerance=gradient_tolerance,
                )
            ),
        )


def _validate_tolerance(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("atol must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError("atol must be a real finite scalar")
    if scalar < 0.0:
        raise ValueError("atol must be greater than or equal to zero")


def _coerce_tolerance(value: float) -> float:
    _validate_tolerance(value)
    return float(value)


def _normalize_qsvt_phases(
    phases: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    if isinstance(phases, np.ndarray):
        flattened = phases.reshape(-1).tolist()
    elif isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        flattened = list(phases)
    else:
        raise TypeError("phases must be a one-dimensional real sequence")

    normalized: list[float] = []
    for value in flattened:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("phases must contain real finite scalars")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise TypeError("phases must contain real finite scalars")
        normalized.append(scalar)
    return tuple(normalized)


def _resolve_expected_matrix(
    *,
    block_encoding: BlockEncoding,
    expected_matrix: npt.NDArray[np.complex128] | None,
) -> npt.NDArray[np.complex128]:
    if expected_matrix is None:
        if not hasattr(block_encoding, "matrix"):
            raise ValueError(
                "expected_matrix must be provided when block_encoding does not "
                "expose a matrix attribute"
            )
        candidate = block_encoding.matrix
    else:
        candidate = expected_matrix

    if not isinstance(candidate, np.ndarray):
        raise TypeError("expected_matrix must be a numpy.ndarray")

    normalized: npt.NDArray[np.complex128] = np.asarray(
        candidate,
        dtype=np.complex128,
    ).copy()
    if normalized.shape != block_encoding.matrix_shape:
        raise ValueError("expected_matrix shape must match block_encoding.matrix_shape")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("expected_matrix must contain only finite entries")
    return normalized


def _project_standard_block(
    circuit: Circuit,
    *,
    rows: int,
    columns: int,
) -> npt.NDArray[np.complex128]:
    matrix = np.asarray(circuit.to_matrix(), dtype=np.complex128)
    return np.asarray(matrix[:rows, :columns], dtype=np.complex128)
