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

"""Private validation and snapshot helpers for Core Hamiltonians."""

from __future__ import annotations

import math

from cqlib.qis import Hamiltonian, PauliString, Phase

_HERMITIAN_RESIDUAL_ATOL = 1e-12


def _is_finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def aggregate_finite_pauli_terms(
    hamiltonian: Hamiltonian,
    *,
    name: str,
) -> tuple[tuple[PauliString, complex], ...]:
    """Validate and stably aggregate canonical finite Pauli terms."""
    coefficients = aggregate_finite_pauli_coefficients(hamiltonian, name=name)
    representatives: dict[tuple[int, int, int], PauliString] = {}
    for pauli, _ in hamiltonian.terms:
        key = (pauli.num_qubits, pauli.x_mask, pauli.z_mask)
        if key not in representatives:
            representative = pauli.copy()
            representative.phase = Phase.plus()
            representatives[key] = representative
    return tuple(
        (representatives[key], coefficient) for key, coefficient in coefficients.items()
    )


def aggregate_finite_pauli_coefficients(
    hamiltonian: Hamiltonian,
    *,
    name: str,
) -> dict[tuple[int, int, int], complex]:
    """Validate and return canonical finite Pauli coefficients."""
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(f"{name} must be a Hamiltonian")

    coefficients: dict[tuple[int, int, int], complex] = {}
    for pauli, coefficient in hamiltonian.terms:
        effective = complex(coefficient) * pauli.phase.to_complex()
        if not _is_finite_complex(effective):
            raise ValueError(f"{name} coefficients must be finite")

        key = (pauli.num_qubits, pauli.x_mask, pauli.z_mask)
        combined = coefficients.get(key, 0.0j) + effective
        if not _is_finite_complex(combined):
            raise ValueError(f"{name} coefficients must be finite")
        coefficients[key] = combined
    return coefficients


def validate_hermitian_observable(
    observable: Hamiltonian | PauliString,
    *,
    name: str,
) -> None:
    """Validate a finite Hermitian Hamiltonian or standalone Pauli string."""
    if isinstance(observable, PauliString):
        phase = observable.phase.to_complex()
        if phase not in (1.0 + 0.0j, -1.0 + 0.0j):
            raise ValueError(f"{name} PauliString phase must be +1 or -1")
        return
    if not isinstance(observable, Hamiltonian):
        raise TypeError(f"{name} must be a Hamiltonian or PauliString")

    aggregated = aggregate_finite_pauli_coefficients(observable, name=name)
    if any(
        coefficient != 0.0j and coefficient.imag != 0.0
        for coefficient in aggregated.values()
    ):
        raise ValueError(f"{name} must be Hermitian")


def validate_nonzero_hermitian_hamiltonian(
    hamiltonian: Hamiltonian,
    *,
    name: str,
) -> None:
    """Validate a finite nonzero Hermitian Hamiltonian."""
    aggregated = aggregate_finite_pauli_coefficients(hamiltonian, name=name)
    if any(
        coefficient != 0.0j and coefficient.imag != 0.0
        for coefficient in aggregated.values()
    ):
        raise ValueError(f"{name} is not Hermitian")
    if not any(coefficient != 0.0j for coefficient in aggregated.values()):
        raise ValueError(f"{name} must not be empty after simplification")


def snapshot_nonzero_hermitian_hamiltonian(
    hamiltonian: Hamiltonian,
    *,
    name: str,
) -> Hamiltonian:
    """Return a defensive finite nonzero Hermitian Hamiltonian snapshot."""
    validate_nonzero_hermitian_hamiltonian(hamiltonian, name=name)
    snapshot = hamiltonian.copy()
    snapshot.simplify()
    if snapshot.num_terms == 0:
        raise ValueError(f"{name} must not be empty after simplification")
    return snapshot


def snapshot_real_hamiltonian(
    mapped: Hamiltonian,
    *,
    expected_width: int,
    name: str,
) -> Hamiltonian:
    """Return an exactly Hermitian mapped snapshot after roundoff removal."""
    if not isinstance(mapped, Hamiltonian):
        raise TypeError("mapper must return a Hamiltonian")
    if mapped.num_qubits != expected_width:
        raise ValueError("mapped Hamiltonian width must match mapper num_qubits")

    terms = aggregate_finite_pauli_terms(mapped, name="mapped Hamiltonian")
    snapshot = Hamiltonian(expected_width)
    for pauli, coefficient in terms:
        if abs(coefficient.imag) > _HERMITIAN_RESIDUAL_ATOL:
            raise ValueError(f"mapped {name} is not Hermitian")
        if coefficient.real != 0.0:
            snapshot.add_term(pauli, coefficient.real)
    return snapshot


__all__: list[str] = []
