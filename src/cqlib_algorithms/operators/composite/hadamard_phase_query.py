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

"""Hadamard phase-query circuit construction.

References
----------
[1] D. Deutsch and R. Jozsa, "Rapid solution of problems by quantum
    computation," Proceedings of the Royal Society A, 439, 553-558, 1992.
    DOI: 10.1098/rspa.1992.0167.
[2] E. Bernstein and U. Vazirani, "Quantum complexity theory," SIAM Journal
    on Computing, 26(5), 1411-1473, 1997.
    DOI: 10.1137/S0097539796300921.
"""

from __future__ import annotations

import copy

from cqlib import Circuit

from ..._utils.circuit import validate_bound_static_unitary_circuit
from ..oracle import BooleanFunction, BooleanVectorFunction, PhaseOracle


class HadamardPhaseQuery:
    """Compose a phase oracle between two all-qubit Hadamard layers.

    The constructed circuit is ``H**n -> S_f -> H**n``, where ``S_f`` is a
    :class:`PhaseOracle`. Scalar and vector-function mask semantics are
    therefore exactly those of the underlying oracle. The component performs
    no measurement, sampling, or promise validation and can be reused by any
    algorithm based on this query transform.

    Args:
        function: Immutable scalar or vector Boolean function, or a phase-oracle
            component implementing that function.
        output_mask: Selected output bits for vector functions, as in
            :class:`PhaseOracle`. An explicit mask must match a supplied oracle.

    Raises:
        TypeError: If the function or mask has the wrong type.
        ValueError: If the output mask is invalid for the function.
    """

    __slots__ = ("_oracle",)

    def __init__(
        self,
        function: BooleanFunction | BooleanVectorFunction | PhaseOracle,
        *,
        output_mask: int | None = None,
    ) -> None:
        if isinstance(function, PhaseOracle):
            if output_mask is not None:
                if isinstance(output_mask, bool) or not isinstance(output_mask, int):
                    raise TypeError("output_mask must be an integer")
                if output_mask != function.output_mask:
                    raise ValueError(
                        "output_mask must match the configured phase oracle"
                    )
            self._oracle = copy.deepcopy(function)
        else:
            self._oracle = PhaseOracle(function, output_mask=output_mask)

    @property
    def function(self) -> BooleanFunction | BooleanVectorFunction:
        """Return the immutable Boolean function used by the phase oracle."""
        return self._oracle.function

    @property
    def output_mask(self) -> int:
        """Return the canonical output mask used by the phase oracle."""
        return self._oracle.output_mask

    @property
    def num_qubits(self) -> int:
        """Return the query-register width."""
        return self.function.num_variables

    def build_circuit(self) -> Circuit:
        """Build a fresh query from the snapshotted, fixed-width unitary oracle."""
        oracle = self._oracle.build_circuit()
        if not isinstance(oracle, Circuit):
            raise TypeError("phase-oracle component must build a Circuit")
        if oracle.num_qubits != self.num_qubits:
            raise ValueError("phase-oracle circuit width must match its function")
        validate_bound_static_unitary_circuit(oracle, "phase-oracle circuit")
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        circuit.compose(oracle, circuit.qubits)
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        return circuit
