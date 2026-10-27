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

import copy
import math
from numbers import Complex
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy, PauliEvolutionAnsatz
from cqlib.qis import (
    FermionicLadder,
    FermionicOp,
    Hamiltonian,
    JordanWignerMapper,
)

from ._validation import (
    copy_hermitian_hamiltonian,
    validate_hermitian_hamiltonian,
    validate_positive_qubit_count,
)
from .base_ansatz import BaseAnsatz


def _snapshot_hopping(value: complex) -> complex:
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise TypeError("hopping must be a complex number")
    hopping = complex(value)
    if not math.isfinite(hopping.real) or not math.isfinite(hopping.imag):
        raise ValueError("hopping must be finite")
    if hopping == 0.0:
        raise ValueError("hopping must be nonzero")
    return hopping


def _validate_mode(mode: int, *, num_modes: int, name: str) -> None:
    if type(mode) is not int:
        raise TypeError(f"{name} must be an int")
    if not 0 <= mode < num_modes:
        raise ValueError(f"{name} is out of range")


def _hopping_operator(
    num_modes: int,
    source_mode: int,
    target_mode: int,
    hopping: complex,
) -> FermionicOp:
    return FermionicOp.from_terms(
        num_modes,
        (
            (
                (
                    FermionicLadder.create(source_mode),
                    FermionicLadder.annihilate(target_mode),
                ),
                hopping,
            ),
            (
                (
                    FermionicLadder.create(target_mode),
                    FermionicLadder.annihilate(source_mode),
                ),
                hopping.conjugate(),
            ),
        ),
    )


def _build_exact_commuting_evolution(
    operator: Hamiltonian,
    *,
    parameter_name: str,
) -> Circuit:
    evolution = (
        PauliEvolutionAnsatz(operator)
        .with_strategy(copy.copy(EvolutionStrategy.exact()))
        .with_time_param_name(parameter_name)
    )
    evolution.validate()
    return evolution.build_circuit(parameter_name)


def _build_hopping_evolution(
    operator: Hamiltonian,
    *,
    source_mode: int,
    target_mode: int,
    phase: float,
    parameter_name: str,
) -> Circuit:
    circuit = Circuit(operator.num_qubits)
    if phase != 0.0:
        circuit.rz(source_mode, -phase / 2.0)
        circuit.rz(target_mode, phase / 2.0)
    evolution = _build_exact_commuting_evolution(
        operator,
        parameter_name=parameter_name,
    )
    circuit.compose(evolution, circuit.qubits)
    if phase != 0.0:
        circuit.rz(source_mode, phase / 2.0)
        circuit.rz(target_mode, -phase / 2.0)
    return circuit


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

    __slots__ = (
        "_hopping",
        "_num_modes",
        "_operator",
        "_phase",
        "_source_mode",
        "_target_mode",
    )

    def __init__(
        self,
        num_modes: int,
        source_mode: int,
        target_mode: int,
        hopping: complex = 1.0,
    ) -> None:
        validate_positive_qubit_count(num_modes)
        _validate_mode(source_mode, num_modes=num_modes, name="source_mode")
        _validate_mode(target_mode, num_modes=num_modes, name="target_mode")
        if source_mode == target_mode:
            raise ValueError("source_mode and target_mode must be distinct")
        frozen_hopping = _snapshot_hopping(hopping)
        real_operator = _hopping_operator(
            num_modes,
            source_mode,
            target_mode,
            complex(abs(frozen_hopping)),
        )
        mapped = JordanWignerMapper().map_operator(real_operator)
        operator = copy_hermitian_hamiltonian(mapped, name="mapped hopping")
        if not operator.all_terms_commute():
            raise ValueError("mapped real hopping terms must commute")

        self._num_modes = num_modes
        self._source_mode = source_mode
        self._target_mode = target_mode
        self._hopping = frozen_hopping
        self._phase = math.atan2(frozen_hopping.imag, frozen_hopping.real)
        self._operator = operator
        self.validate()

    @property
    def hopping(self) -> complex:
        """Return the fixed directed hopping coefficient."""
        return self._hopping

    @property
    def modes(self) -> tuple[int, int]:
        """Return ``(source_mode, target_mode)``."""
        return (self._source_mode, self._target_mode)

    def _validate_configuration(self) -> None:
        validate_positive_qubit_count(self._num_modes)
        _validate_mode(
            self._source_mode,
            num_modes=self._num_modes,
            name="source_mode",
        )
        _validate_mode(
            self._target_mode,
            num_modes=self._num_modes,
            name="target_mode",
        )
        if self._source_mode == self._target_mode:
            raise ValueError("source_mode and target_mode must be distinct")
        _snapshot_hopping(self._hopping)
        validate_hermitian_hamiltonian(self._operator, name="mapped hopping")
        if not self._operator.all_terms_commute():
            raise ValueError("mapped real hopping terms must commute")

    def _build_for_parameter(self, parameter_name: str) -> Circuit:
        return _build_hopping_evolution(
            self._operator,
            source_mode=self._source_mode,
            target_mode=self._target_mode,
            phase=self._phase,
            parameter_name=parameter_name,
        )

    def _build_circuit(self, prefix: str) -> Circuit:
        return self._build_for_parameter(f"{prefix}_0")

    def _parameter_count(self) -> int:
        return 1

    def _qubit_count(self) -> int:
        return self._num_modes


__all__ = ["FermionicHoppingAnsatz"]
