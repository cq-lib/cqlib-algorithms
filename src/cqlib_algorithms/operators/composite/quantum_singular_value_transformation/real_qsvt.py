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

r"""Real-valued quantum singular value transformation via phase symmetrisation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cqlib import Circuit

from ...oracle.block_encoding import BlockEncoding
from ...oracle.unitary_matrix_oracle import UnitaryMatrixOracle
from ._phase_solver import convert_qsp_phases as _convert_qsp_phases
from ._phase_solver import polynomial_to_chebyshev
from .qsp import QuantumSignalProcessing
from .qsvt import QuantumSingularValueTransformation


class RealValuedQSVT:
    r"""Real-valued QSVT obtained by symmetrising ``U_{\Phi}`` and ``U_{-\Phi}``.

    The standard QSVT core :class:`QuantumSingularValueTransformation`
    implements a complex polynomial ``P`` whose real part equals the target
    real polynomial ``P_{\mathrm{Re}}`` (Corollary 10 in Ref. [1]).
    Concretely, on the singular-value basis the projected block of the core
    circuit acts as

    .. math::

        \tilde{\Pi} U_{\Phi} \tilde{\Pi}
            = P(\sigma) \, |u\rangle\langle v|,
        \qquad
        \mathrm{Re}\bigl(P(\sigma)\bigr) = P_{\mathrm{Re}}(\sigma).

    To recover the full real polynomial rather than only its real part, note
    that the complex conjugate of ``P`` is realised by negating every phase in
    the sequence:

    .. math::

        P(\sigma) = U_{\Phi},
        \qquad
        P^{*}(\sigma) = U_{-\Phi},
        \qquad
        P_{\mathrm{Re}}(\sigma) = \frac{P(\sigma) + P^{*}(\sigma)}{2}
            = \frac{U_{\Phi} + U_{-\Phi}}{2}.

    This averaging is performed coherently with an auxiliary real-part ancilla
    qubit ``a`` placed at the most-significant position.  Starting from
    ``|+\rangle_a = (|0\rangle + |1\rangle)/\sqrt{2}``, the controlled
    block-diagonal operator

    .. math::

        C = |0\rangle\langle 0| \otimes U_{\Phi}
            + |1\rangle\langle 1| \otimes U_{-\Phi}

    maps the ancilla register to
    ``(|0\rangle U_{\Phi} + |1\rangle U_{-\Phi})/\sqrt{2}``.  A final
    Hadamard on the ancilla produces

    .. math::

        \frac{1}{2}\Bigl[
            |0\rangle\, (U_{\Phi} + U_{-\Phi})
            + |1\rangle\, (U_{\Phi} - U_{-\Phi})
        \Bigr],

    so projecting ``a`` onto ``|0\rangle`` yields
    ``(U_{\Phi} + U_{-\Phi})/2``, which is exactly ``P_{\mathrm{Re}}``
    (Corollary 18 in Ref. [1]).

    The assembled circuit is therefore

    .. math::

        H_a \, C \, H_a.

    Parameters
    ----------
    block_encoding
        Standard block encoding used as the QSVT signal operator.
    phases
        QSVT phase sequence ``\Phi`` for the positive branch.  The negative
        branch uses ``-\Phi`` (every phase negated, order preserved).
    degree
        Polynomial degree of the target transformation.  When *None*
        (default), inferred from ``len(phases)``.

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
    """

    __slots__ = (
        "_block_encoding",
        "_degree",
        "_negative_core",
        "_phases",
        "_positive_core",
    )

    _block_encoding: BlockEncoding
    _degree: int
    _phases: tuple[float, ...]
    _positive_core: QuantumSingularValueTransformation
    _negative_core: QuantumSingularValueTransformation

    _SYNTHESIS_ATOL: float = 1e-7

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
            raise ValueError("RealValuedQSVT requires a standard block encoding")

        self._block_encoding = block_encoding
        self._phases = tuple(float(p) for p in np.asarray(phases).reshape(-1))
        self._degree = degree if degree is not None else len(self._phases)

        negative_phases = tuple(-p for p in self._phases)
        self._positive_core = QuantumSingularValueTransformation(
            block_encoding,
            self._phases,
            degree=self._degree,
        )
        self._negative_core = QuantumSingularValueTransformation(
            block_encoding,
            negative_phases,
            degree=self._degree,
        )

    @property
    def block_encoding(self) -> BlockEncoding:
        r"""Return the underlying standard block encoding."""

        return self._block_encoding

    @property
    def phases(self) -> tuple[float, ...]:
        r"""Return the immutable QSVT phase sequence of the positive branch."""

        return self._phases

    @property
    def degree(self) -> int:
        r"""Return the polynomial degree of the target transformation."""

        return self._degree

    @property
    def num_qubits(self) -> int:
        r"""Return the total qubit width including the real-part ancilla."""

        return self._positive_core.num_qubits + 1

    @property
    def real_ancilla_index(self) -> int:
        r"""Return the local index of the real-part ancilla qubit."""

        return self._positive_core.num_qubits

    def build_circuit(self) -> Circuit:
        r"""Build and return the symmetrised real-valued QSVT circuit.

        The circuit contains an additional real-part ancilla at the
        most-significant position.  Projecting that ancilla onto
        ``|0\rangle`` together with the block-encoding ancilla yields the
        real polynomial evaluated on the singular values of the encoded
        matrix.
        """

        circuit = Circuit(self.num_qubits)
        ancilla = self.real_ancilla_index
        circuit.h(ancilla)
        circuit.compose(self._build_controlled_oracle())
        circuit.h(ancilla)
        return circuit

    def _build_controlled_oracle(self) -> Circuit:
        r"""Return the controlled multiplexer circuit.

        Implements :math:`C = |0\rangle\langle 0| \otimes U_{\Phi}
        + |1\rangle\langle 1| \otimes U_{-\Phi}` on the real-part ancilla
        and the system qubits using two controlled gates interleaved with
        two X gates on the ancilla:

        .. math::

            C = (X_a \, \mathrm{c}(U_{-\Phi}) \, X_a) \, \mathrm{c}(U_{\Phi}),

        where :math:`\mathrm{c}(U)` denotes ``U`` controlled on
        :math:`|1\rangle_a`.  The outer :math:`X_a` pair turns the second
        controlled gate into a control-on-:math:`|0\rangle` gate, so the
        two controlled operators act on complementary ancilla states.
        """

        core_dimension = 2**self._positive_core.num_qubits
        positive_matrix = np.asarray(
            self._positive_core.build_circuit().to_matrix(),
            dtype=np.complex128,
        )
        negative_matrix = np.asarray(
            self._negative_core.build_circuit().to_matrix(),
            dtype=np.complex128,
        )
        # Control-on-|1> unitaries: diag(I, U) with the ancilla as MSB.
        controlled_positive = np.block(
            [
                [
                    np.eye(core_dimension, dtype=np.complex128),
                    np.zeros((core_dimension, core_dimension), dtype=np.complex128),
                ],
                [
                    np.zeros((core_dimension, core_dimension), dtype=np.complex128),
                    positive_matrix,
                ],
            ]
        )
        controlled_negative = np.block(
            [
                [
                    np.eye(core_dimension, dtype=np.complex128),
                    np.zeros((core_dimension, core_dimension), dtype=np.complex128),
                ],
                [
                    np.zeros((core_dimension, core_dimension), dtype=np.complex128),
                    negative_matrix,
                ],
            ]
        )

        positive_oracle = UnitaryMatrixOracle(
            controlled_positive,
            atol=self._SYNTHESIS_ATOL,
            validate=True,
        )
        negative_oracle = UnitaryMatrixOracle(
            controlled_negative,
            atol=self._SYNTHESIS_ATOL,
            validate=True,
        )

        ancilla = self.real_ancilla_index
        core_qubits = list(range(self._positive_core.num_qubits))
        # The oracle's last qubit is the MSB (control ancilla); map it onto
        # the real-part ancilla and the rest onto the core register.
        controlled_qubits = [*core_qubits, ancilla]

        circuit = Circuit(self.num_qubits)
        circuit.x(ancilla)
        circuit.compose(positive_oracle.build_circuit(), qubits=controlled_qubits)
        circuit.x(ancilla)
        circuit.compose(negative_oracle.build_circuit(), qubits=controlled_qubits)
        return circuit

    @classmethod
    def from_chebyshev_coefficients(
        cls,
        block_encoding: BlockEncoding,
        coefficients: Sequence[float] | np.ndarray,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> RealValuedQSVT:
        r"""Solve QSP phases for a Chebyshev-basis real polynomial and bind them.

        ``P(x)`` is specified by its Chebyshev-basis coefficients
        ``P(x) = \sum_k c_k T_k(x)``.  The resulting circuit implements the
        real polynomial ``P`` itself through ``U_\Phi / U_{-\Phi}``
        symmetrisation (Corollary 18 in Ref. [1]).

        Parameters
        ----------
        block_encoding
            Standard block encoding used as the QSVT signal operator.
        coefficients
            Chebyshev-basis coefficients ``(c_0, c_1, ..., c_d)`` in ascending
            order.
        tolerance
            Convergence tolerance for the QSP phase solver.
        max_iterations
            Maximum iterations for the QSP phase solver.
        gradient_tolerance
            Gradient convergence tolerance for the QSP phase solver.

        Returns
        -------
        RealValuedQSVT
            A bound real-valued QSVT instance ready for circuit construction.

        Raises
        ------
        ValueError
            If a degree-0 coefficient lies outside ``[-1, 1]``.
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
    ) -> RealValuedQSVT:
        r"""Solve QSP phases for a power-basis real polynomial and bind them.

        ``P(x)`` is specified by its standard power (monomial) coefficients
        ``P(x) = a_0 + a_1 x + ... + a_d x^d``.  The coefficients are first
        converted to the Chebyshev basis before solving.

        Parameters
        ----------
        block_encoding
            Standard block encoding used as the QSVT signal operator.
        coefficients
            Power-basis coefficients ``(a_0, a_1, ..., a_d)`` in ascending
            order.
        tolerance
            Convergence tolerance for the QSP phase solver.
        max_iterations
            Maximum iterations for the QSP phase solver.
        gradient_tolerance
            Gradient convergence tolerance for the QSP phase solver.

        Returns
        -------
        RealValuedQSVT
            A bound real-valued QSVT instance ready for circuit construction.
        """

        return cls.from_chebyshev_coefficients(
            block_encoding,
            polynomial_to_chebyshev(coefficients),
            tolerance=tolerance,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )
