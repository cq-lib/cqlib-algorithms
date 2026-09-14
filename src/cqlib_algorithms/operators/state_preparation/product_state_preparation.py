"""Unentangled product-state preparation."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit

from ...protocols.state_preparation import StatePreparation
from ._state_synthesis import (
    ComplexVector,
    require_normalized,
    validate_prefix,
)


@final
class ProductStatePreparation(StatePreparation):
    r"""Prepare :math:`\bigotimes_q(a_q|0\rangle+b_q|1\rangle)`.

    Item ``q`` of ``qubit_states`` configures qubit ``q``. Every item must
    contain exactly two finite amplitudes and have unit norm within 1e-12.
    Accepted roundoff is corrected before the copied configuration is stored.
    """

    __slots__ = ("_qubit_states",)

    def __init__(self, qubit_states: Sequence[Sequence[complex]]) -> None:
        if isinstance(qubit_states, (str, bytes)) or not isinstance(
            qubit_states, Sequence
        ):
            raise TypeError("qubit_states must be a sequence")
        if not qubit_states:
            raise ValueError("qubit_states must not be empty")

        copied: list[ComplexVector] = []
        for qubit, state in enumerate(qubit_states):
            if isinstance(state, (str, bytes)) or not isinstance(state, Sequence):
                raise TypeError(f"qubit_states[{qubit}] must be a sequence")
            if len(state) != 2:
                raise ValueError(f"qubit_states[{qubit}] must contain two amplitudes")
            values: list[complex] = []
            for basis, raw_value in enumerate(state):
                try:
                    value = complex(raw_value)
                except TypeError as error:
                    raise TypeError(
                        f"qubit_states[{qubit}][{basis}] must be convertible to complex"
                    ) from error
                except ValueError as error:
                    raise ValueError(
                        f"qubit_states[{qubit}][{basis}] is not a valid complex value"
                    ) from error
                if not math.isfinite(value.real) or not math.isfinite(value.imag):
                    raise ValueError(f"qubit_states[{qubit}][{basis}] must be finite")
                values.append(value)
            copied.append(
                require_normalized(tuple(values), name=f"qubit_states[{qubit}]")
            )
        self._qubit_states = tuple(copied)

    def validate(self) -> None:
        """Validate every immutable single-qubit state."""
        for qubit, state in enumerate(self._qubit_states):
            require_normalized(state, name=f"qubit_states[{qubit}]")

    def build_circuit(self, prefix: str) -> Circuit:
        """Return a new circuit of independent RY and Phase preparations."""
        validate_prefix(prefix)
        circuit = Circuit(len(self._qubit_states))
        global_phase = 0.0
        for qubit, (zero_amplitude, one_amplitude) in enumerate(self._qubit_states):
            zero_magnitude = abs(zero_amplitude)
            one_magnitude = abs(one_amplitude)
            theta = 2.0 * math.atan2(one_magnitude, zero_magnitude)
            reference_phase = (
                cmath.phase(zero_amplitude)
                if zero_amplitude != 0.0
                else cmath.phase(one_amplitude)
            )
            relative_phase = (
                cmath.phase(one_amplitude) - reference_phase
                if one_amplitude != 0.0
                else 0.0
            )
            if theta != 0.0:
                circuit.ry(qubit, theta)
            if relative_phase != 0.0:
                circuit.phase(qubit, relative_phase)
            global_phase += reference_phase
        circuit.set_global_phase(global_phase)
        return circuit

    def num_parameters(self) -> int:
        """Return zero because every product-state factor is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the number of single-qubit factors."""
        return len(self._qubit_states)
