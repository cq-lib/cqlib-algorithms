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

"""Phase-oracle circuit construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cqlib import Circuit
from cqlib.circuit.gates import StandardGate

from .boolean_function import BooleanFunction
from .boolean_vector_function import BooleanVectorFunction


@dataclass(frozen=True, slots=True, init=False)
class PhaseOracle:
    """Construct a scalar or output-mask parity phase oracle.

    A scalar ``f`` retains ``S_f|x> = (-1)**f(x)|x>`` with the implicit mask
    one. For vector ``F``, the exact phase is
    ``(-1)**parity(output_mask & F_int(x))``. This is parity, not logical OR.
    Input qubit ``q_i`` represents ``function.variable_order[i]`` and bit ``i``
    of the computational-basis integer, so ``q0`` is the LSB. No output qubits
    or ancillas are present.

    Args:
        function: Immutable scalar or vector Boolean function to mark.
        output_mask: Selected output bits. Scalar functions default to one and
            accept no other value. Vector functions require an explicit mask.

    Raises:
        TypeError: If the function or mask has the wrong type.
        ValueError: If the mask is missing or outside its valid range.
    """

    function: BooleanFunction | BooleanVectorFunction
    output_mask: int

    def __init__(
        self,
        function: BooleanFunction | BooleanVectorFunction,
        *,
        output_mask: int | None = None,
    ) -> None:
        if not isinstance(function, (BooleanFunction, BooleanVectorFunction)):
            raise TypeError(
                "function must be a BooleanFunction or BooleanVectorFunction"
            )
        if output_mask is not None and (
            isinstance(output_mask, bool) or not isinstance(output_mask, int)
        ):
            raise TypeError("output_mask must be an integer")

        if isinstance(function, BooleanFunction):
            mask = 1 if output_mask is None else output_mask
            if mask != 1:
                raise ValueError("output_mask must be 1 for a BooleanFunction")
        else:
            if output_mask is None:
                raise ValueError("output_mask is required for a BooleanVectorFunction")
            mask = output_mask
            if not 1 <= mask < 1 << function.num_outputs:
                raise ValueError("output_mask must select one or more existing outputs")

        object.__setattr__(self, "function", function)
        object.__setattr__(self, "output_mask", mask)

    def build_circuit(self) -> Circuit:
        """Build and return a fresh phase-oracle circuit."""
        num_qubits = self.function.num_variables
        circuit = Circuit(num_qubits)
        if isinstance(self.function, BooleanFunction):
            marked_states = self.function.marked_states
        else:
            marked_states = tuple(
                state
                for state, row in enumerate(self.function.truth_table)
                if sum(
                    int(value)
                    for output, value in enumerate(row)
                    if (self.output_mask >> output) & 1
                )
                % 2
            )

        if len(marked_states) == 1 << num_qubits:
            circuit.set_global_phase(math.pi)
            return circuit

        for state in marked_states:
            zero_qubits = [
                qubit for qubit in range(num_qubits) if not (state >> qubit) & 1
            ]
            for qubit in zero_qubits:
                circuit.x(qubit)
            if num_qubits == 1:
                circuit.z(0)
            else:
                circuit.append_mc_gate(
                    StandardGate.Z.control(num_qubits - 1),
                    list(range(num_qubits)),
                )
            for qubit in reversed(zero_qubits):
                circuit.x(qubit)

        return circuit
