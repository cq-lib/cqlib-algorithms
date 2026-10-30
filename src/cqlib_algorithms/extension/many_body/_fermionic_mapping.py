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

"""Shared fermion-to-qubit mapping for extension operators."""

from __future__ import annotations

from cqlib.qis import (
    FermionicOp,
    FermionToQubitMapper,
    Hamiltonian,
    JordanWignerMapper,
)

from ._mapping_validation import snapshot_real_hamiltonian


def map_fermionic_operator(
    operator: FermionicOp,
    mapper: FermionToQubitMapper | None,
    *,
    name: str,
) -> Hamiltonian:
    """Map one fermionic operator with strict width and Hermiticity checks."""
    if not isinstance(operator, FermionicOp):
        raise TypeError("operator must be a FermionicOp")
    selected_mapper = JordanWignerMapper() if mapper is None else mapper
    if not isinstance(selected_mapper, FermionToQubitMapper):
        raise TypeError("mapper must implement FermionToQubitMapper")
    expected_width = selected_mapper.num_qubits(operator.num_modes)
    if type(expected_width) is not int:
        raise TypeError("mapper num_qubits must return an int")
    if expected_width < 1:
        raise ValueError("mapper num_qubits must return a positive width")
    mapped = selected_mapper.map_operator(operator)
    return snapshot_real_hamiltonian(
        mapped,
        expected_width=expected_width,
        name=name,
    )


__all__: list[str] = []
