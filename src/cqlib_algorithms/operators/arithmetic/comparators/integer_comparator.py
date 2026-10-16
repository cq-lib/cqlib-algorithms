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
"""Integer comparator circuits.

Overview
--------
This module implements a minimal integer comparator against a classical
constant. The circuit flips a flag qubit when an input register satisfies a
selected relation with respect to the compiled integer value.

Notation
--------
The input register uses LSB-first indexing. Qubit 0 stores the least
significant bit of the integer x.

The local layout is

    |x>_n |flag> |helper>

and the action is

    |x>|flag> -> |x>|flag XOR predicate(x, value)>

References
----------
[1] S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton,
    "A New Quantum Ripple Carry Addition Circuit," 2004.
    arXiv:quant-ph/0410184.
[2] T. G. Draper, "Addition on a Quantum Computer," 2000.
    arXiv:quant-ph/0008033.
"""

from __future__ import annotations

from typing import Literal

from cqlib.circuit.gates import StandardGate

from cqlib import Circuit
from cqlib_algorithms.operators.arithmetic.adders.constant_adder import (
    DraperConstantAdder,
)

ComparatorRelation = Literal["geq", "lt"]
ComparatorSynthesis = Literal["ripple_carry", "draper_qft"]

_VALID_RELATIONS: frozenset[str] = frozenset({"geq", "lt"})
_VALID_SYNTHESIS: frozenset[str] = frozenset({"ripple_carry", "draper_qft"})


def _build_draper_geq_comparator(
    circuit: Circuit,
    x_indices: list[int],
    flag_index: int,
    helper_index: int,
    value: int,
) -> Circuit:
    """Build the Draper branch for x >= value."""
    n = len(x_indices)
    offset = (1 << n) - value
    adder = DraperConstantAdder(n + 1, offset)
    adder_circuit = adder.build_circuit()
    work_wires = list(x_indices) + [helper_index]

    circuit.compose(adder_circuit, work_wires)
    circuit.cx(helper_index, flag_index)
    circuit.compose(adder_circuit.inverse(), work_wires)
    return circuit


def _build_draper_lt_comparator(
    circuit: Circuit,
    x_indices: list[int],
    flag_index: int,
    helper_index: int,
    value: int,
) -> Circuit:
    """Build the Draper branch for x < value."""
    n = len(x_indices)
    offset = (1 << n) - value
    adder = DraperConstantAdder(n + 1, offset)
    adder_circuit = adder.build_circuit()
    work_wires = list(x_indices) + [helper_index]

    circuit.compose(adder_circuit, work_wires)
    circuit.x(helper_index)
    circuit.cx(helper_index, flag_index)
    circuit.x(helper_index)
    circuit.compose(adder_circuit.inverse(), work_wires)
    return circuit


def _build_ripple_geq_comparator(
    circuit: Circuit,
    x_indices: list[int],
    flag_index: int,
    value: int,
) -> Circuit:
    """Build the ripple branch for x >= value."""
    n = len(x_indices)
    negative_controls = [
        x_indices[bit]
        for bit in range(n)
        if ((value >> bit) & 1) == 0
    ]
    for qubit in negative_controls:
        circuit.x(qubit)
    equality_gate = StandardGate.X.control(n)
    circuit.append_mc_gate(
        equality_gate,
        list(x_indices) + [flag_index],
    )
    for qubit in reversed(negative_controls):
        circuit.x(qubit)

    for bit in range(n - 1, -1, -1):
        if (value >> bit) & 1:
            continue
        controls = [x_indices[idx] for idx in range(n - 1, bit, -1)]
        controls.append(x_indices[bit])
        open_controls = [
            x_indices[idx]
            for idx in range(n - 1, bit, -1)
            if ((value >> idx) & 1) == 0
        ]
        for qubit in open_controls:
            circuit.x(qubit)
        if controls:
            gate = StandardGate.X.control(len(controls))
            circuit.append_mc_gate(gate, controls + [flag_index])
        else:
            circuit.x(flag_index)
        for qubit in reversed(open_controls):
            circuit.x(qubit)
    return circuit


def _build_ripple_lt_comparator(
    circuit: Circuit,
    x_indices: list[int],
    flag_index: int,
    value: int,
) -> Circuit:
    """Build the ripple branch for x < value."""
    n = len(x_indices)
    for bit in range(n - 1, -1, -1):
        if ((value >> bit) & 1) == 0:
            continue
        controls = [x_indices[idx] for idx in range(n - 1, bit, -1)]
        controls.append(x_indices[bit])
        open_controls = [
            x_indices[idx]
            for idx in range(n - 1, bit, -1)
            if ((value >> idx) & 1) == 0
        ]
        open_controls.append(x_indices[bit])
        for qubit in open_controls:
            circuit.x(qubit)
        if controls:
            gate = StandardGate.X.control(len(controls))
            circuit.append_mc_gate(gate, controls + [flag_index])
        else:
            circuit.x(flag_index)
        for qubit in reversed(open_controls):
            circuit.x(qubit)
    return circuit


class IntegerComparator:
    """Minimal value-based integer comparator."""

    def __init__(
        self,
        num_state_qubits: int,
        value: int,
        *,
        relation: ComparatorRelation = "geq",
        synthesis: ComparatorSynthesis = "ripple_carry",
        name: str | None = None,
    ) -> None:
        if not isinstance(num_state_qubits, int):
            raise TypeError(
                "num_state_qubits must be an integer, got "
                f"{num_state_qubits!r}."
            )
        if num_state_qubits < 1:
            raise ValueError(
                f"num_state_qubits must be >= 1, got {num_state_qubits!r}."
            )
        if not isinstance(value, int):
            raise TypeError(f"value must be an integer, got {value!r}.")
        if relation not in _VALID_RELATIONS:
            raise ValueError(
                f"Unknown comparator relation {relation!r}. "
                f"Expected one of {sorted(_VALID_RELATIONS)}."
            )
        if synthesis not in _VALID_SYNTHESIS:
            raise ValueError(
                f"Unknown comparator synthesis {synthesis!r}. "
                f"Expected one of {sorted(_VALID_SYNTHESIS)}."
            )
        if name is not None and not isinstance(name, str):
            raise TypeError(
                "name must be a string or None, got "
                f"{name!r}."
            )

        modulus = 1 << int(num_state_qubits)
        if not 0 <= value < modulus:
            raise ValueError(
                f"value must satisfy 0 <= value < 2**num_state_qubits, got {value!r}."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._value = int(value)
        self._relation = relation
        self._synthesis = synthesis
        self._num_qubits = self._num_state_qubits + 2
        self._x_register_indices = list(range(self._num_state_qubits))
        self._flag_index = self._num_state_qubits
        self._helper_index = self._num_state_qubits + 1
        self._name = name or "IntegerComparator"

    @property
    def name(self) -> str:
        """Return the configured circuit name."""
        return self._name

    @property
    def num_state_qubits(self) -> int:
        """Return the width of the input register."""
        return self._num_state_qubits

    @property
    def num_qubits(self) -> int:
        """Return the total local circuit width."""
        return self._num_qubits

    @property
    def value(self) -> int:
        """Return the compiled comparison value."""
        return self._value

    @property
    def relation(self) -> ComparatorRelation:
        """Return the selected comparison relation."""
        return self._relation

    @property
    def synthesis(self) -> ComparatorSynthesis:
        """Return the selected synthesis strategy."""
        return self._synthesis

    @property
    def x_register_indices(self) -> list[int]:
        """Return local wire indices of the compared register."""
        return list(self._x_register_indices)

    @property
    def flag_index(self) -> int:
        """Return the local index of the output flag qubit."""
        return self._flag_index

    @property
    def helper_index(self) -> int:
        """Return the local index of the reserved helper qubit."""
        return self._helper_index

    def build_circuit(self) -> Circuit:
        """Build a comparator circuit for the configured relation."""
        circuit = Circuit(self._num_qubits)
        x_indices = self._x_register_indices
        flag_index = self._flag_index
        helper_index = self._helper_index

        if self._synthesis == "draper_qft":
            if self._relation == "geq":
                return _build_draper_geq_comparator(
                    circuit,
                    x_indices,
                    flag_index,
                    helper_index,
                    self._value,
                )
            if self._relation == "lt":
                return _build_draper_lt_comparator(
                    circuit,
                    x_indices,
                    flag_index,
                    helper_index,
                    self._value,
                )
            raise RuntimeError(
                f"Unsupported comparator relation {self._relation!r}."
            )

        if self._synthesis == "ripple_carry":
            if self._relation == "geq":
                return _build_ripple_geq_comparator(
                    circuit,
                    x_indices,
                    flag_index,
                    self._value,
                )

            if self._relation == "lt":
                return _build_ripple_lt_comparator(
                    circuit,
                    x_indices,
                    flag_index,
                    self._value,
                )

            raise RuntimeError(
                f"Unsupported comparator relation {self._relation!r}."
            )

        raise RuntimeError(
            f"Unsupported comparator synthesis {self._synthesis!r}."
        )

