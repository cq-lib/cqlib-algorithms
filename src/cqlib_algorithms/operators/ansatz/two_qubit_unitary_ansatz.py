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

"""Provide fixed-width parameterizations of two-qubit unitary blocks.

References
----------
[1] S. Joshi, *piQture*, GitHub repository, 2024,
    https://github.com/SaashaJoshi/piQture.
"""

from __future__ import annotations

import math
from typing import Literal

from cqlib.circuit import Circuit, Parameter

from .base_ansatz import BaseAnsatz

_Parameterization = Literal["simple", "general"]
_Structure = Literal["real", "complex"]


class TwoQubitUnitaryAnsatz(BaseAnsatz):
    """Build a two-qubit trainable unitary block in one of four modes.

    ``simple`` selects a two-parameter low-depth block and is not a universal
    two-qubit parameterization. ``general`` with ``structure="real"`` spans
    the six-dimensional ``SO(4)`` family. ``general`` with
    ``structure="complex"`` has 15 parameters and covers arbitrary two-qubit
    unitaries modulo global phase. Qubit 0 is the least-significant basis bit;
    CX directions are part of the fixed architecture and are documented in
    the user guide.

    Args:
        parameterization: Low-depth ``"simple"`` or expressive ``"general"``.
        structure: Real-matrix ``"real"`` or complex-unitary ``"complex"``.

    Raises:
        TypeError: If either option is not a string.
        ValueError: If either option is unsupported.
    """

    __slots__ = ("_parameterization", "_structure")

    def __init__(
        self,
        *,
        parameterization: _Parameterization = "general",
        structure: _Structure = "complex",
    ) -> None:
        _validate_choice(
            parameterization,
            "parameterization",
            ("simple", "general"),
        )
        _validate_choice(structure, "structure", ("real", "complex"))
        self._parameterization = parameterization
        self._structure = structure
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the frozen mode selection."""
        _validate_choice(
            self._parameterization,
            "parameterization",
            ("simple", "general"),
        )
        _validate_choice(self._structure, "structure", ("real", "complex"))

    def _build_circuit(self, prefix: str) -> Circuit:
        """Build the selected fixed two-qubit architecture."""
        parameters = [
            Parameter(f"{prefix}_{index}") for index in range(self._parameter_count())
        ]
        if self._parameterization == "simple":
            return self._build_simple(parameters)
        if self._structure == "real":
            return self._build_general_real(parameters)
        return self._build_general_complex(parameters)

    def _build_simple(self, parameters: list[Parameter]) -> Circuit:
        circuit = Circuit(2)
        if self._structure == "real":
            circuit.ry(0, parameters[0])
        else:
            circuit.rx(0, parameters[0])
        circuit.ry(1, parameters[1])
        circuit.cx(0, 1)
        return circuit

    @staticmethod
    def _build_general_real(parameters: list[Parameter]) -> Circuit:
        circuit = Circuit(2)
        circuit.rz(0, math.pi / 2)
        circuit.rz(1, math.pi / 2)
        circuit.ry(1, math.pi / 2)
        circuit.cx(1, 0)

        circuit.rz(0, parameters[0])
        circuit.ry(0, parameters[1])
        circuit.rz(0, parameters[2])
        circuit.rz(1, parameters[3])
        circuit.ry(1, parameters[4])
        circuit.rz(1, parameters[5])

        circuit.cx(1, 0)
        circuit.ry(1, -math.pi / 2)
        circuit.rz(0, -math.pi / 2)
        circuit.rz(1, -math.pi / 2)
        return circuit

    @staticmethod
    def _build_general_complex(parameters: list[Parameter]) -> Circuit:
        circuit = Circuit(2)
        circuit.rz(0, parameters[0])
        circuit.ry(0, parameters[1])
        circuit.rz(0, parameters[2])
        circuit.rz(1, parameters[3])
        circuit.ry(1, parameters[4])
        circuit.rz(1, parameters[5] + math.pi / 2)
        circuit.cx(1, 0)

        circuit.rz(0, 2 * parameters[6] - math.pi / 2)
        circuit.ry(1, math.pi / 2 - 2 * parameters[7])
        circuit.cx(0, 1)
        circuit.ry(1, 2 * parameters[8] - math.pi / 2)

        circuit.cx(1, 0)
        circuit.rz(1, parameters[9])
        circuit.ry(1, parameters[10])
        circuit.rz(1, parameters[11])
        circuit.rz(0, parameters[12] - math.pi / 2)
        circuit.ry(0, parameters[13])
        circuit.rz(0, parameters[14])
        return circuit

    def _parameter_count(self) -> int:
        """Return the number of independent trainable angles."""
        if self._parameterization == "simple":
            return 2
        return 6 if self._structure == "real" else 15

    def _qubit_count(self) -> int:
        """Return the fixed two-qubit width."""
        return 2


def _validate_choice(value: object, name: str, choices: tuple[str, ...]) -> None:
    """Validate one string-valued architecture option."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        expected = " or ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be {expected}")
