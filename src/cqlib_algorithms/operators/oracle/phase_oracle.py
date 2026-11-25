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
from collections.abc import Mapping, Sequence

from cqlib import Circuit
from cqlib.circuit.gates import StandardGate

from ..._utils.numeric import validate_positive_integer
from .boolean_function import BooleanFunction
from .boolean_vector_function import BooleanVectorFunction


def _mark_bit_pattern(
    circuit: Circuit, positions: Sequence[int], zeros: Sequence[int]
) -> None:
    """Negate the basis states matching the selected bits."""
    if not positions:
        circuit.set_global_phase(math.pi)
        return
    for qubit in zeros:
        circuit.x(qubit)
    if len(positions) == 1:
        circuit.z(positions[0])
    else:
        circuit.append_mc_gate(
            StandardGate.Z.control(len(positions) - 1), list(positions)
        )
    for qubit in reversed(zeros):
        circuit.x(qubit)


class BitPatternPhaseOracle:
    """Flip the phase when all specified qubits match their assigned bits.

    Args:
        num_qubits: Positive register width.
        bits: Qubit positions mapped to zero or one. Other qubits are free.
            An empty mapping negates every basis state.

    Raises:
        TypeError: If the width, mapping, positions, or bits have invalid types.
        ValueError: If a position is out of range or a value is nonbinary.
    """

    __slots__ = ("_num_qubits", "_positions", "_zeros")

    def __init__(self, num_qubits: int, bits: Mapping[int, int | bool]) -> None:
        validate_positive_integer(num_qubits, "num_qubits")
        if not isinstance(bits, Mapping):
            raise TypeError("bits must be a mapping")
        positions = []
        zeros = []
        for position, bit in bits.items():
            if isinstance(position, bool) or not isinstance(position, int):
                raise TypeError("bit positions must be integers")
            if not 0 <= position < num_qubits:
                raise ValueError("bit position is outside the register")
            if not isinstance(bit, int):
                raise TypeError("bit values must be binary integers or bool")
            if bit not in (0, 1):
                raise ValueError("bit values must be zero or one")
            positions.append(position)
            if bit == 0:
                zeros.append(position)
        self._num_qubits = num_qubits
        self._positions = tuple(positions)
        self._zeros = tuple(zeros)

    @property
    def num_qubits(self) -> int:
        """Return the full register width."""
        return self._num_qubits

    @property
    def marked_states(self) -> tuple[int, ...]:
        """Enumerate marked indices on demand, with one entry per free-bit assignment."""
        fixed = set(self._positions)
        ones = fixed.difference(self._zeros)
        states = [sum(1 << position for position in ones)]
        for position in range(self.num_qubits):
            if position not in fixed:
                states += [state | (1 << position) for state in states]
        return tuple(states)

    def build_circuit(self) -> Circuit:
        """Build the phase mark using gates on the specified bits only."""
        circuit = Circuit(self.num_qubits)
        _mark_bit_pattern(circuit, self._positions, self._zeros)
        return circuit


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

    __slots__ = ("_function", "_output_mask")

    _function: BooleanFunction | BooleanVectorFunction
    _output_mask: int

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

        self._function = function
        self._output_mask = mask

    @property
    def function(self) -> BooleanFunction | BooleanVectorFunction:
        """Return the immutable Boolean function marked by the oracle."""
        return self._function

    @property
    def output_mask(self) -> int:
        """Return the selected output-bit mask."""
        return self._output_mask

    @property
    def num_qubits(self) -> int:
        """Return the number of input variables."""
        return self.function.num_variables

    @property
    def marked_states(self) -> tuple[int, ...]:
        """Return the ordered basis indices negated by the selected predicate."""
        if isinstance(self.function, BooleanFunction):
            return self.function.marked_states
        return tuple(
            state
            for state, row in enumerate(self.function.truth_table)
            if sum(
                int(value)
                for output, value in enumerate(row)
                if (self.output_mask >> output) & 1
            )
            % 2
        )

    def build_circuit(self) -> Circuit:
        """Build and return a fresh phase-oracle circuit."""
        num_qubits = self.function.num_variables
        circuit = Circuit(num_qubits)
        marked_states = self.marked_states
        if len(marked_states) == 1 << num_qubits:
            circuit.set_global_phase(math.pi)
            return circuit

        for state in marked_states:
            zero_qubits = [
                qubit for qubit in range(num_qubits) if not (state >> qubit) & 1
            ]
            _mark_bit_pattern(circuit, range(num_qubits), zero_qubits)

        return circuit
