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

"""Reference and second-order initial points for ordered UCCSD circuits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType

import numpy as np

from ..fermionic.excitation_generation import generate_uccsd_excitations
from .integrals import ElectronicIntegrals
from .system import ElectronicSystem

_CANONICAL_ATOL = 1e-8
_DENOMINATOR_ATOL = 1e-12


def _parameter_names(system: ElectronicSystem, prefix: str) -> tuple[str, ...]:
    if not isinstance(system, ElectronicSystem):
        raise TypeError("system must be an ElectronicSystem")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str")
    excitations = generate_uccsd_excitations(system.occupation)
    if not excitations:
        raise ValueError("system must produce at least one UCCSD excitation")
    return tuple(f"{prefix}_theta_{index}" for index in range(len(excitations)))


def hartree_fock_initial_point(
    system: ElectronicSystem,
    *,
    prefix: str = "ucc",
) -> Mapping[str, float]:
    """Return the all-zero point that leaves the UCCSD reference unchanged.

    Args:
        system: Electronic system defining the UCCSD excitation pool.
        prefix: Prefix used by the corresponding UCCSD circuit.

    Returns:
        Immutable mapping from every UCCSD parameter to zero.

    Raises:
        TypeError: If ``system`` or ``prefix`` has the wrong type.
        ValueError: If the system has no UCCSD excitations.
    """
    return MappingProxyType(dict.fromkeys(_parameter_names(system, prefix), 0.0))


def _canonical_fock_matrix(system: ElectronicSystem) -> np.ndarray:
    if system.is_spin_resolved:
        raise ValueError("MP2 initialization does not support spin-resolved integrals")
    occupation = system.occupation
    if occupation.num_alpha != occupation.num_beta:
        raise ValueError("MP2 initialization requires a closed-shell occupation")
    if system.orbital_energies is None:
        raise ValueError("MP2 initialization requires orbital_energies")

    integrals = system.integrals
    assert isinstance(integrals, ElectronicIntegrals)
    fock = np.array(integrals.one_body, copy=True)
    for occupied in range(occupation.num_alpha):
        fock += 2.0 * integrals.two_body[:, :, occupied, occupied]
        fock -= integrals.two_body[:, occupied, occupied, :]
    expected = np.diag(system.orbital_energies)
    if not np.allclose(fock, expected, rtol=0.0, atol=_CANONICAL_ATOL):
        raise ValueError(
            "MP2 initialization requires canonical restricted-spin orbitals"
        )
    return fock


def _spin_orbital_integral(
    system: ElectronicSystem,
    target_a: int,
    target_b: int,
    source_i: int,
    source_j: int,
) -> float:
    integrals = system.integrals
    assert isinstance(integrals, ElectronicIntegrals)
    tensor = integrals.two_body
    a, spin_a = divmod(target_a, 2)
    b, spin_b = divmod(target_b, 2)
    i, spin_i = divmod(source_i, 2)
    j, spin_j = divmod(source_j, 2)
    direct = tensor[a, i, b, j] if (spin_a, spin_b) == (spin_i, spin_j) else 0.0
    exchange = tensor[a, j, b, i] if (spin_a, spin_b) == (spin_j, spin_i) else 0.0
    return float(direct - exchange)


def mp2_initial_point(
    system: ElectronicSystem,
    *,
    prefix: str = "ucc",
    threshold: float = 1e-12,
) -> Mapping[str, float]:
    """Return canonical restricted-spin MP2 doubles as UCCSD parameters.

    Singles are zero by the Brillouin condition. A double excitation ``ij->ab``
    receives ``<ab||ij> / (epsilon_i + epsilon_j - epsilon_a - epsilon_b)``.
    Values with absolute magnitude at most ``threshold`` are set to zero.

    Args:
        system: Closed-shell restricted system in canonical orbitals.
        prefix: Prefix used by the corresponding UCCSD circuit.
        threshold: Non-negative magnitude below which amplitudes become zero.

    Returns:
        Immutable mapping of UCCSD symbols to singles-zero MP2 amplitudes.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the system is incompatible with canonical restricted
            MP2 initialization or contains a zero energy denominator.
    """
    if not isinstance(system, ElectronicSystem):
        raise TypeError("system must be an ElectronicSystem")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be finite")
    if threshold_value < 0.0:
        raise ValueError("threshold must be non-negative")
    _canonical_fock_matrix(system)
    assert system.orbital_energies is not None

    excitations = generate_uccsd_excitations(system.occupation)
    if not excitations:
        raise ValueError("system must produce at least one UCCSD excitation")
    spin_energies = np.repeat(system.orbital_energies, 2)
    parameters: dict[str, float] = {}
    for index, excitation in enumerate(excitations):
        value = 0.0
        if excitation.rank == 2:
            source_i, source_j = excitation.sources
            target_a, target_b = excitation.targets
            denominator = float(
                spin_energies[source_i]
                + spin_energies[source_j]
                - spin_energies[target_a]
                - spin_energies[target_b]
            )
            if abs(denominator) <= _DENOMINATOR_ATOL:
                raise ValueError("MP2 excitation energy denominator is zero")
            value = (
                _spin_orbital_integral(
                    system,
                    target_a,
                    target_b,
                    source_i,
                    source_j,
                )
                / denominator
            )
            if abs(value) <= threshold_value:
                value = 0.0
        parameters[f"{prefix}_theta_{index}"] = value
    return MappingProxyType(parameters)


__all__ = ["hartree_fock_initial_point", "mp2_initial_point"]
