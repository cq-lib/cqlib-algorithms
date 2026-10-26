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

r"""Standard quantum singular value transformation circuit construction.

References
----------
[1] A. Gilyen, Y. Su, G. H. Low, and N. Wiebe, "Quantum singular value transformation
    and beyond: exponential improvements for quantum matrix arithmetics," Proceedings of
    the 51st Annual ACM SIGACT Symposium on Theory of Computing, pp. 193-204, 2019.
    arXiv:1806.01838.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Literal

import numpy as np
import numpy.typing as npt

from cqlib import Circuit

from ...oracle.block_encoding import BlockEncoding
from ._phase_solver import (
    convert_qsp_phases as _convert_qsp_phases,
    polynomial_to_chebyshev,
    solve_haah_phases,
)
from ._phase_solver._complex_solver import _validate_complex_qsp_constraints
from .projector_controlled_phase import ProjectorControlledPhase
from .qsp import QuantumSignalProcessing


class QuantumSingularValueTransformation:
    r"""Build the standard QSVT alternating sequence from a phase sequence.

    This implementation currently supports only standard block encodings with

    .. math::

        \Pi = \tilde{\Pi} = |0\rangle\langle 0|^{\otimes a} \otimes I.

    The alternating sequence follows Definition 15 in Ref. [1].  Given a
    QSVT phase sequence ``(phi_1, ..., phi_d)`` the circuit assembles the
    ``W``-form product of block-encoding ``U``/``U^\dagger`` and
    projector-controlled phase gates.

    The projected block implements a complex polynomial ``P`` whose real
    part equals the target real polynomial ``P_Re`` (Corollary 10 in
    Ref. [1]).  Obtaining the full real polynomial itself requires the
    additional ``U_\Phi / U_{-\Phi}`` symmetrisation of Corollary 18,
    which is outside the scope of this class.

    Args:
        block_encoding: Standard block encoding used as the QSVT signal operator.
        phases: QSVT phase sequence used by the alternating construction.
        degree: Polynomial degree of the target transformation.  When *None*
            (default), inferred from ``len(phases)``.

    Raises:
        TypeError: If ``block_encoding`` or ``phases`` has an invalid type.
        ValueError: If ``block_encoding`` is not standard.
    """

    __slots__ = (
        "_block_encoding",
        "_degree",
        "_left_projector",
        "_phases",
        "_right_projector",
    )

    _block_encoding: BlockEncoding
    _degree: int
    _phases: tuple[float, ...]
    _right_projector: ProjectorControlledPhase
    _left_projector: ProjectorControlledPhase

    def __init__(
        self,
        block_encoding: BlockEncoding,
        phases: Sequence[float] | np.ndarray,
        *,
        degree: int | None = None,
    ) -> None:
        if not isinstance(block_encoding, BlockEncoding):
            raise TypeError("block_encoding must be a BlockEncoding")
        if not block_encoding.is_standard:
            raise ValueError("QSVT currently supports only standard block encodings")

        self._block_encoding = block_encoding
        self._phases = _normalize_qsvt_phases(phases)
        self._degree = degree if degree is not None else len(self._phases)
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
        r"""Return the underlying standard block encoding."""

        return self._block_encoding

    @property
    def phases(self) -> tuple[float, ...]:
        r"""Return the immutable QSVT phase sequence."""

        return self._phases

    @property
    def degree(self) -> int:
        r"""Return the polynomial degree of the target transformation."""

        return self._degree

    @property
    def right_projector(self) -> ProjectorControlledPhase:
        r"""Return the standard right-projector phase template."""

        return self._right_projector

    @property
    def left_projector(self) -> ProjectorControlledPhase:
        r"""Return the standard left-projector phase template."""

        return self._left_projector

    @property
    def num_qubits(self) -> int:
        r"""Return the width of the local QSVT circuit."""

        return self.block_encoding.total_qubits

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh local-numbering QSVT circuit."""

        circuit = Circuit(self.num_qubits)
        phase_count = len(self.phases)

        if phase_count == 0:
            return circuit

        if self._degree == 0:
            circuit.compose(
                self.left_projector.with_phase(self.phases[0]).build_circuit()
            )
            return circuit

        for reverse_index, phase in enumerate(reversed(self.phases), start=1):
            phase_index = phase_count - reverse_index + 1
            if phase_index % 2 == phase_count % 2:
                circuit.compose(self.block_encoding.build_circuit())
                circuit.compose(self.left_projector.with_phase(phase).build_circuit())
            else:
                circuit.compose(self.block_encoding.build_adjoint_circuit())
                circuit.compose(self.right_projector.with_phase(phase).build_circuit())
        return circuit

    def check_block_encoding(
        self,
        *,
        expected_matrix: npt.NDArray[np.complex128] | None = None,
        atol: float = 1e-10,
    ) -> float:
        r"""Validate the standard projected identity ``A = PiTilde U Pi``.

        Args:
            expected_matrix: Explicit target matrix ``A``. When omitted, the method uses the
                ``matrix`` attribute of the current block encoding if available.
            atol: Absolute tolerance for the projected-block comparison.

        Returns:
            Frobenius norm of the projected-block reconstruction error.

        Raises:
            TypeError: If the expected matrix or tolerance has an invalid type.
            ValueError: If no explicit target matrix is available, if the target matrix has
                an invalid shape, or if the projected identity is not satisfied.

        """

        tolerance = _normalize_tolerance(atol)
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
        r"""Solve QSP phases for a Chebyshev-basis real polynomial and bind them.

        ``P(x)`` is specified by its Chebyshev-basis coefficients
        ``P(x) = \sum_k c_k T_k(x)``.  The resulting QSVT circuit implements a
        complex polynomial whose real part equals ``P`` (Corollary 10 in
        Ref. [1]).

        Args:
            block_encoding: Standard block encoding used as the QSVT signal operator.
            coefficients: Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
                order.
            tolerance: Convergence tolerance for the QSP phase solver.
            max_iterations: Maximum iterations for the QSP phase solver.
            gradient_tolerance: Gradient convergence tolerance for the QSP phase solver.

        Returns:
            A bound QSVT instance ready for circuit construction.

        Raises:
            ValueError: If a degree-0 coefficient lies outside ``[-1, 1]``.

        """

        coeffs = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        degree = int(len(coeffs)) - 1

        if degree == 0:
            return cls(
                block_encoding,
                (math.acos(float(coeffs[0])),),
                degree=0,
            )

        return cls(
            block_encoding,
            _convert_qsp_phases(
                QuantumSignalProcessing.solve_phases(
                    coefficients,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    gradient_tolerance=gradient_tolerance,
                )
            ),
            degree=degree,
        )

    @classmethod
    def from_polynomial_coefficients(
        cls,
        block_encoding: BlockEncoding,
        coefficients: Sequence[float] | np.ndarray,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSingularValueTransformation:
        r"""Solve QSP phases for a power-basis real polynomial and bind them.

        ``P(x)`` is specified by its standard power (monomial) coefficients
        ``P(x) = a_0 + a_1 x + ... + a_d x^d``.  The coefficients are first
        converted to the Chebyshev basis before solving.

        Args:
            block_encoding: Standard block encoding used as the QSVT signal operator.
            coefficients: Power-basis coefficients ``(a_0, a_1, ..., a_d)`` in ascending
                order.
            tolerance: Convergence tolerance for the QSP phase solver.
            max_iterations: Maximum iterations for the QSP phase solver.
            gradient_tolerance: Gradient convergence tolerance for the QSP phase solver.

        Returns:
            A bound QSVT instance ready for circuit construction.

        """
        return cls.from_chebyshev_coefficients(
            block_encoding,
            polynomial_to_chebyshev(coefficients),
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )

    @classmethod
    def from_complex_chebyshev_coefficients(
        cls,
        block_encoding: BlockEncoding,
        coefficients: Sequence[complex] | np.ndarray,
        *,
        synthesis: Literal["bfgs", "haah"] = "bfgs",
        tolerance: float | None = None,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSingularValueTransformation:
        r"""Solve QSP phases for a complex Chebyshev polynomial and bind them.

        ``P(x)`` is specified by its complex Chebyshev-basis coefficients
        ``P(x) = \sum_k c_k T_k(x)``.  The resulting QSVT circuit implements
        the full complex polynomial ``P`` on the singular values of the
        block-encoded matrix (Corollary 8 / Theorem 17 in Ref. [1]).

        Args:
            block_encoding: Standard block encoding used as the QSVT signal operator.
            coefficients: Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
                order.  May be complex.
            synthesis: Phase-synthesis backend.  ``"bfgs"`` fits the QSP response with
                the quasi-Newton solver; ``"haah"`` uses the deterministic
                product decomposition of Haah (arXiv:1806.10236).
            tolerance: Backend tolerance.  When *None* the backend default is used:
                ``1e-12`` for ``"bfgs"`` (fitted mean-squared loss) and ``1e-10``
                for ``"haah"`` (peeling tolerance).
            max_iterations: Maximum iterations for the ``"bfgs"`` backend.  Ignored by
                ``"haah"``.
            gradient_tolerance: Gradient convergence tolerance for the ``"bfgs"`` backend.
                Ignored by ``"haah"``.

        Returns:
            A bound QSVT instance ready for circuit construction.

        Raises:
            ValueError: If ``synthesis`` is not a supported backend, if the polynomial
                violates the QSP parity or boundedness constraints, or if a
                degree-0 coefficient does not lie on the unit circle.
            RuntimeError: If the phase solver fails numerically.

        """
        coeffs = np.asarray(coefficients).reshape(-1)
        degree = int(len(coeffs)) - 1

        if degree == 0:
            return cls(
                block_encoding,
                (_complex_degree_zero_phase(coeffs[0]),),
                degree=0,
            )

        return cls(
            block_encoding,
            _convert_qsp_phases(
                _solve_complex_phases(
                    coefficients,
                    synthesis=synthesis,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    gradient_tolerance=gradient_tolerance,
                )
            ),
            degree=degree,
        )

    @classmethod
    def from_complex_polynomial_coefficients(
        cls,
        block_encoding: BlockEncoding,
        coefficients: Sequence[complex] | np.ndarray,
        *,
        synthesis: Literal["bfgs", "haah"] = "bfgs",
        tolerance: float | None = None,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> QuantumSingularValueTransformation:
        r"""Solve QSP phases for a complex power-basis polynomial and bind them.

        ``P(x)`` is specified by its standard power (monomial) coefficients
        ``P(x) = a_0 + a_1 x + ... + a_d x^d``.  The coefficients are first
        converted to the Chebyshev basis before solving.

        Args:
            block_encoding: Standard block encoding used as the QSVT signal operator.
            coefficients: Power-basis coefficients ``(a_0, a_1, ..., a_d)`` in ascending
                order.  May be complex.
            synthesis: Phase-synthesis backend, forwarded to
                :meth:`from_complex_chebyshev_coefficients`.
            tolerance: Backend tolerance.  When *None* the backend default is used.
            max_iterations: Maximum iterations for the ``"bfgs"`` backend.  Ignored by
                ``"haah"``.
            gradient_tolerance: Gradient convergence tolerance for the ``"bfgs"`` backend.
                Ignored by ``"haah"``.

        Returns:
            A bound QSVT instance ready for circuit construction.

        """
        return cls.from_complex_chebyshev_coefficients(
            block_encoding,
            polynomial_to_chebyshev(coefficients),
            synthesis=synthesis,
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )


def _solve_complex_phases(
    coefficients: Sequence[complex] | np.ndarray,
    *,
    synthesis: Literal["bfgs", "haah"],
    tolerance: float | None,
    max_iterations: int,
    gradient_tolerance: float,
) -> np.ndarray:
    r"""Dispatch a complex target to the requested phase-synthesis backend.

    The ``"bfgs"`` backend fits the complex QSP response with a quasi-Newton
    search and defaults to a mean-squared loss tolerance of ``1e-12``; the
    ``"haah"`` backend factors the response deterministically and defaults to
    a peeling tolerance of ``1e-10``.
    """
    if synthesis == "haah":
        if tolerance is None:
            return solve_haah_phases(coefficients)
        return solve_haah_phases(coefficients, tolerance=tolerance)

    if synthesis == "bfgs":
        return QuantumSignalProcessing.solve_complex_phases(
            coefficients,
            tolerance=1e-12 if tolerance is None else tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )

    raise ValueError("synthesis must be 'bfgs' or 'haah'")


def _complex_degree_zero_phase(coefficient: complex) -> float:
    r"""Return the single QSVT phase realising a degree-0 complex polynomial.

    For an even-degree target the alternating sequence reduces to a lone
    projector-controlled phase gate, whose projected block is ``e^{i\phi}``
    on the encoded subspace.  A constant target ``P(x) = c_0`` is therefore
    realisable exactly when ``|c_0| = 1``; Corollary 8 conditions 2 and 3 pin
    any constant polynomial onto the unit circle.  The phase is then
    ``arg(c_0)``, so no phase solver is invoked.

    Args:
        coefficient: Constant Chebyshev coefficient ``c_0`` of the target polynomial.

    Returns:
        The QSVT phase ``arg(c_0)``.

    Raises:
        ValueError: If ``c_0`` violates the Corollary 8 magnitude conditions, i.e. does
            not lie on the unit circle.

    """
    constant = complex(coefficient)
    _validate_complex_qsp_constraints(
        np.asarray([constant], dtype=np.complex128),
    )
    return math.atan2(constant.imag, constant.real)


def _validate_tolerance(value: float) -> None:
    r"""Validate a non-negative finite tolerance."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("atol must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError("atol must be a real finite scalar")
    if scalar < 0.0:
        raise ValueError("atol must be greater than or equal to zero")


def _normalize_tolerance(value: float) -> float:
    r"""Validate and return a tolerance in normalized form."""
    _validate_tolerance(value)
    return float(value)


def _normalize_qsvt_phases(
    phases: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    r"""Validate and return a QSVT phase sequence in normalized form."""
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
    r"""Resolve the explicit target matrix, falling back to ``block_encoding.matrix``."""
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
    r"""Return the top-left ``rows x columns`` projected block of ``circuit``."""
    matrix = np.asarray(circuit.to_matrix(), dtype=np.complex128)
    return np.asarray(matrix[:rows, :columns], dtype=np.complex128)
