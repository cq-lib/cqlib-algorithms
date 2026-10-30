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

"""Shared validation for mapped many-body Hamiltonians."""

from __future__ import annotations

import math

from cqlib.qis import Hamiltonian, PauliString, Phase

_HERMITIAN_RESIDUAL_ATOL = 1e-12


def _normalized_pauli(pauli: PauliString) -> PauliString:
    """Return the same Pauli word with unit phase."""
    normalized = pauli.copy()
    normalized.phase = Phase.plus()
    return normalized


def snapshot_real_hamiltonian(
    mapped: Hamiltonian,
    *,
    expected_width: int,
    name: str,
) -> Hamiltonian:
    """Return an exactly Hermitian snapshot after bounded roundoff removal."""
    if not isinstance(mapped, Hamiltonian):
        raise TypeError("mapper must return a Hamiltonian")
    if mapped.num_qubits != expected_width:
        raise ValueError("mapped Hamiltonian width must match mapper num_qubits")

    aggregated: dict[tuple[int, int, int], tuple[PauliString, complex]] = {}
    for pauli, coefficient in mapped.terms:
        effective = complex(coefficient) * pauli.phase.to_complex()
        if not math.isfinite(effective.real) or not math.isfinite(effective.imag):
            raise ValueError("mapped Hamiltonian coefficients must be finite")
        key = (pauli.num_qubits, pauli.x_mask, pauli.z_mask)
        normalized = _normalized_pauli(pauli)
        combined = aggregated.get(key, (normalized, 0.0j))[1] + effective
        if not math.isfinite(combined.real) or not math.isfinite(combined.imag):
            raise ValueError("mapped Hamiltonian coefficients must be finite")
        aggregated[key] = (normalized, combined)

    snapshot = Hamiltonian(expected_width)
    for pauli, coefficient in aggregated.values():
        if abs(coefficient.imag) > _HERMITIAN_RESIDUAL_ATOL:
            raise ValueError(f"mapped {name} is not Hermitian")
        if coefficient.real != 0.0:
            snapshot.add_term(pauli, coefficient.real)
    return snapshot


__all__: list[str] = []
