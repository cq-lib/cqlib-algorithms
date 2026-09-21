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

"""Bit-flip-oracle circuit construction."""

from __future__ import annotations

from cqlib.circuit.gates import StandardGate

from cqlib import Circuit

from .boolean_function import BooleanFunction
from .boolean_vector_function import BooleanVectorFunction


class BitFlipOracle:
    """Construct ``U_F|x>|y> = |x>|y xor F(x)>`` without ancillas.

    The input register occupies qubits ``q0`` through ``q(n-1)`` with ``q0``
    as the LSB. Output ``i`` occupies ``q(n+i)``. Scalar functions retain one
    output on ``q(n)``.

    Args:
        function: Immutable scalar or vector Boolean function to compute.

    Raises:
        TypeError: If ``function`` is not a supported Boolean function.
    """

    __slots__ = ("_function",)

    _function: BooleanFunction | BooleanVectorFunction

    def __init__(self, function: BooleanFunction | BooleanVectorFunction) -> None:
        if not isinstance(function, (BooleanFunction, BooleanVectorFunction)):
            raise TypeError(
                "function must be a BooleanFunction or BooleanVectorFunction"
            )
        self._function = function

    @property
    def function(self) -> BooleanFunction | BooleanVectorFunction:
        """Return the immutable Boolean function computed by the oracle."""
        return self._function

    def build_circuit(self) -> Circuit:
        """Build and return a fresh bit-flip-oracle circuit."""
        num_inputs = self.function.num_variables
        outputs = (
            (self.function,)
            if isinstance(self.function, BooleanFunction)
            else self.function.outputs
        )
        circuit = Circuit(num_inputs + len(outputs))
        controlled_x = StandardGate.X.control(num_inputs)

        for output_index, output_function in enumerate(outputs):
            output_qubit = num_inputs + output_index
            marked_states = output_function.marked_states
            if len(marked_states) == 1 << num_inputs:
                circuit.x(output_qubit)
                continue
            for state in marked_states:
                zero_qubits = [
                    qubit for qubit in range(num_inputs) if not (state >> qubit) & 1
                ]
                for qubit in zero_qubits:
                    circuit.x(qubit)
                circuit.append_mc_gate(controlled_x, [*range(num_inputs), output_qubit])
                for qubit in reversed(zero_qubits):
                    circuit.x(qubit)

        return circuit
