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

"""Ordered factorized unitary coupled-cluster ansatz."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.circuit import Circuit
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import FermionToQubitMapper, Hamiltonian, JordanWignerMapper

from ...extension.chemistry import FermionicExcitation, GroupedFermionicGenerator
from ...protocols.state_preparation import StatePreparation
from ._ordered_evolution import (
    build_fixed_reference_evolution,
    validate_fixed_reference_evolution,
)
from ._validation import copy_hermitian_hamiltonian, snapshot_fixed_initial_state
from .base_ansatz import _DelegatingAnsatz
from .evolved_operator_ansatz import EvolvedOperatorAnsatz

FermionicGenerator = FermionicExcitation | GroupedFermionicGenerator


def _snapshot_generators(
    excitations: Sequence[FermionicGenerator],
) -> tuple[FermionicGenerator, ...]:
    """Return a nonempty ordered generator tuple without deduplication."""
    if isinstance(excitations, (str, bytes)) or not isinstance(excitations, Sequence):
        raise TypeError("excitations must be a sequence of fermionic generators")
    snapshot = tuple(excitations)
    if not snapshot:
        raise ValueError("excitations must not be empty")
    if any(
        not isinstance(item, (FermionicExcitation, GroupedFermionicGenerator))
        for item in snapshot
    ):
        raise TypeError(
            "excitations must contain only FermionicExcitation or "
            "GroupedFermionicGenerator values"
        )
    width = snapshot[0].num_spin_orbitals
    if any(item.num_spin_orbitals != width for item in snapshot[1:]):
        raise ValueError("all excitations must use the same number of spin orbitals")
    return snapshot


def _mapped_operator_snapshots(
    excitations: tuple[FermionicGenerator, ...],
    mapper: FermionToQubitMapper,
) -> tuple[Hamiltonian, ...]:
    """Map every excitation once and return exact-evolution-safe snapshots."""
    mapped_width = mapper.num_qubits(excitations[0].num_spin_orbitals)
    if type(mapped_width) is not int:
        raise TypeError("mapper num_qubits must return an int")
    if mapped_width < 1:
        raise ValueError("mapper num_qubits must return a positive width")

    operators: list[Hamiltonian] = []
    for index, excitation in enumerate(excitations):
        mapped = mapper.map_operator(excitation.to_fermionic_generator())
        if not isinstance(mapped, Hamiltonian):
            raise TypeError("mapper must return a Hamiltonian")
        if mapped.num_qubits != mapped_width:
            raise ValueError(
                "mapped generator Hamiltonian width must match mapper num_qubits"
            )
        snapshot = copy_hermitian_hamiltonian(
            mapped,
            name=f"mapped generator[{index}]",
        )
        if not snapshot.all_terms_commute():
            raise ValueError(
                "mapped generator Hamiltonian terms must commute for exact evolution"
            )
        operators.append(snapshot)
    return tuple(operators)


@final
class UCCAnsatz(_DelegatingAnsatz):
    """Build an ordered factorized unitary coupled-cluster circuit.

    Sequence item ``k`` defines the Hermitian generator ``G_k=i(T_k-T_k†)``
    or a fixed real linear combination of such generators.
    The circuit appends ``exp(-i theta_k G_k)`` in input order, so the
    excitation unitary is ``U_(m-1) ... U_0``. With an initial-state circuit
    ``A``, the full matrix is ``U_(m-1) ... U_0 A``; without one, ``A`` is the
    identity. This disentangled product generally differs from the exponential
    of the summed generators, and repeated excitations remain meaningful
    ordered factors. Trainable symbols follow excitation order and are named
    ``{prefix}_theta_0`` through ``{prefix}_theta_(m-1)``.

    Args:
        excitations: Non-empty ordered fermionic generators with equal width.
        mapper: Fermion-to-qubit mapper; Jordan--Wigner when omitted.
        initial_state: Optional fixed state preparation prepended once.

    Raises:
        TypeError: If a generator, mapper, or initial state has the wrong type.
        ValueError: If widths disagree or exact mapped evolution is unavailable.
    """

    __slots__ = ("_initial_circuit",)

    def __init__(
        self,
        excitations: Sequence[FermionicGenerator],
        *,
        mapper: FermionToQubitMapper | None = None,
        initial_state: StatePreparation | None = None,
    ) -> None:
        frozen_excitations = _snapshot_generators(excitations)
        selected_mapper = JordanWignerMapper() if mapper is None else mapper
        if not isinstance(selected_mapper, FermionToQubitMapper):
            raise TypeError("mapper must implement FermionToQubitMapper")
        operators = _mapped_operator_snapshots(frozen_excitations, selected_mapper)
        inner = EvolvedOperatorAnsatz(
            operators,
            strategy=EvolutionStrategy.exact(),
            remove_identity_operators=False,
        )
        self._initial_circuit = snapshot_fixed_initial_state(
            initial_state,
            num_qubits=inner.num_qubits(),
            prefix="ucc_initial",
        )
        super().__init__(inner)
        self.validate()

    def _validate_configuration(self) -> None:
        """Validate the exact evolution and fixed initial-state snapshots."""
        validate_fixed_reference_evolution(self._initial_circuit, self._inner)

    def _build_circuit(self, prefix: str) -> Circuit:
        """Return the fixed initial state followed by ordered excitation factors."""
        return build_fixed_reference_evolution(
            self._initial_circuit,
            self._inner,
            parameter_prefix=f"{prefix}_theta",
        )


__all__ = ["UCCAnsatz"]
