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

"""Exact Jordan--Wigner evolution of one complex fermionic hopping term."""

from __future__ import annotations

from typing import final

from cqlib.circuit import Circuit

from ._fermionic_hopping import _HoppingEvolution
from .base_ansatz import BaseAnsatz


@final
class FermionicHoppingAnsatz(BaseAnsatz):
    r"""Evolve one Hermitian complex hopping factor exactly.

    The generated circuit is

    .. math::

        \exp[-i\theta(t c_p^\dagger c_q+t^*c_q^\dagger c_p)].

    Modes map directly to equally indexed qubits using Jordan--Wigner
    occupation encoding. A fixed number-phase conjugation reduces complex
    ``t`` to a real hopping operator whose two mapped Pauli terms commute;
    the shared symbolic evolution is therefore exact and retains the full
    intervening parity string for non-adjacent modes.

    Args:
        num_modes: Positive number of fermionic modes.
        source_mode: First hopping mode.
        target_mode: Distinct second hopping mode.
        hopping: Finite directed complex hopping coefficient.

    Raises:
        TypeError: If a count, index, or coefficient has the wrong type.
        ValueError: If indices or the mapped exact-evolution factor are invalid.
    """

    __slots__ = ("_evolution",)

    def __init__(
        self,
        num_modes: int,
        source_mode: int,
        target_mode: int,
        hopping: complex = 1.0,
    ) -> None:
        self._evolution = _HoppingEvolution.create(
            num_modes,
            source_mode,
            target_mode,
            hopping,
        )
        self.validate()

    @property
    def hopping(self) -> complex:
        """Return the fixed directed hopping coefficient."""
        return self._evolution.hopping

    @property
    def modes(self) -> tuple[int, int]:
        """Return ``(source_mode, target_mode)``."""
        return (self._evolution.source_mode, self._evolution.target_mode)

    def _validate_configuration(self) -> None:
        self._evolution.validate()

    def _build_circuit(self, prefix: str) -> Circuit:
        return self._evolution.build_circuit(f"{prefix}_0")

    def _parameter_count(self) -> int:
        return 1

    def _qubit_count(self) -> int:
        return self._evolution.num_modes


__all__ = ["FermionicHoppingAnsatz"]
