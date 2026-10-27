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

"""Electronic-energy conversion to Core fermionic and qubit operators."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib.qis import (
    FermionicLadder,
    FermionicOp,
    FermionToQubitMapper,
    Hamiltonian,
)

from ...many_body._fermionic_mapping import map_fermionic_operator
from .integrals import ElectronicIntegrals
from .spin_resolved_integrals import SpinResolvedElectronicIntegrals


@dataclass(frozen=True, slots=True)
class ElectronicEnergy:
    """An electronic Hamiltonian in restricted or unrestricted MO integrals.

    Spin-orbital mode ``2*p`` is alpha and mode ``2*p + 1`` is beta. Restricted
    input shares ``h`` and ``V[p,q,r,s] = (pq|rs)`` between both spins:

    ``c + sum h[p,q] a†[p,s] a[q,s]``
    ``+ 1/2 sum V[p,q,r,s] a†[p,s] a†[r,t] a[s,t] a[q,s]``.

    Spin-resolved input stores separate one-body and same-spin tensors. Its
    alpha-beta tensor multiplies ``a†[p,a] a†[r,b] a[s,b] a[q,a]`` with no
    half factor because the symmetry-equivalent beta-alpha term is implicit.

    Args:
        integrals: Restricted or spin-resolved molecular-orbital integrals.

    Raises:
        TypeError: If ``integrals`` has an unsupported type.
    """

    integrals: ElectronicIntegrals | SpinResolvedElectronicIntegrals

    def __post_init__(self) -> None:
        if not isinstance(
            self.integrals,
            (ElectronicIntegrals, SpinResolvedElectronicIntegrals),
        ):
            raise TypeError(
                "integrals must be ElectronicIntegrals or "
                "SpinResolvedElectronicIntegrals"
            )

    @property
    def num_spin_orbitals(self) -> int:
        """Return the number of interleaved spin-orbital modes."""
        return 2 * self.integrals.num_spatial_orbitals

    def to_fermionic_op(self) -> FermionicOp:
        """Build the exact second-quantized fermionic Hamiltonian.

        Returns:
            Fermionic Hamiltonian in interleaved alpha/beta mode order.
        """
        if isinstance(self.integrals, SpinResolvedElectronicIntegrals):
            return self._spin_resolved_fermionic_op()

        terms: list[tuple[tuple[FermionicLadder, ...], float]] = []
        if self.integrals.constant != 0.0:
            terms.append(((), self.integrals.constant))

        n = self.integrals.num_spatial_orbitals
        for p in range(n):
            for q in range(n):
                coefficient = float(self.integrals.one_body[p, q])
                if coefficient == 0.0:
                    continue
                for spin in range(2):
                    terms.append(
                        (
                            (
                                FermionicLadder.create(2 * p + spin),
                                FermionicLadder.annihilate(2 * q + spin),
                            ),
                            coefficient,
                        )
                    )

        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coefficient = 0.5 * float(self.integrals.two_body[p, q, r, s])
                        if coefficient == 0.0:
                            continue
                        for sigma in range(2):
                            for tau in range(2):
                                terms.append(
                                    (
                                        (
                                            FermionicLadder.create(2 * p + sigma),
                                            FermionicLadder.create(2 * r + tau),
                                            FermionicLadder.annihilate(2 * s + tau),
                                            FermionicLadder.annihilate(2 * q + sigma),
                                        ),
                                        coefficient,
                                    )
                                )
        return FermionicOp.from_terms(self.num_spin_orbitals, terms)

    def _spin_resolved_fermionic_op(self) -> FermionicOp:
        integrals = self.integrals
        assert isinstance(integrals, SpinResolvedElectronicIntegrals)
        terms: list[tuple[tuple[FermionicLadder, ...], float]] = []
        if integrals.constant != 0.0:
            terms.append(((), integrals.constant))

        n = integrals.num_spatial_orbitals
        for p in range(n):
            for q in range(n):
                alpha = float(integrals.one_body_alpha[p, q])
                beta = float(integrals.one_body_beta[p, q])
                if alpha != 0.0:
                    terms.append(
                        (
                            (
                                FermionicLadder.create(2 * p),
                                FermionicLadder.annihilate(2 * q),
                            ),
                            alpha,
                        )
                    )
                if beta != 0.0:
                    terms.append(
                        (
                            (
                                FermionicLadder.create(2 * p + 1),
                                FermionicLadder.annihilate(2 * q + 1),
                            ),
                            beta,
                        )
                    )

        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        same_spin = (
                            0.5 * float(integrals.two_body_alpha_alpha[p, q, r, s]),
                            0.5 * float(integrals.two_body_beta_beta[p, q, r, s]),
                        )
                        for spin, coefficient in enumerate(same_spin):
                            if coefficient == 0.0:
                                continue
                            terms.append(
                                (
                                    (
                                        FermionicLadder.create(2 * p + spin),
                                        FermionicLadder.create(2 * r + spin),
                                        FermionicLadder.annihilate(2 * s + spin),
                                        FermionicLadder.annihilate(2 * q + spin),
                                    ),
                                    coefficient,
                                )
                            )

                        mixed = float(integrals.two_body_alpha_beta[p, q, r, s])
                        if mixed != 0.0:
                            terms.append(
                                (
                                    (
                                        FermionicLadder.create(2 * p),
                                        FermionicLadder.create(2 * r + 1),
                                        FermionicLadder.annihilate(2 * s + 1),
                                        FermionicLadder.annihilate(2 * q),
                                    ),
                                    mixed,
                                )
                            )
        return FermionicOp.from_terms(self.num_spin_orbitals, terms)

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map and snapshot the electronic Hamiltonian.

        Pauli phases and duplicate words are combined before Hermiticity is
        checked. Imaginary coefficient residuals no larger than ``1e-12`` are
        projected to zero because real molecular integrals define a Hermitian
        operator; larger residuals are rejected.

        Args:
            mapper: Fermion-to-qubit mapper. The Jordan--Wigner mapping is used
                when omitted.

        Returns:
            Hermitian qubit Hamiltonian with canonical real coefficients.

        Raises:
            TypeError: If the mapper or mapped result has the wrong type.
            ValueError: If the mapped operator is inconsistent or non-Hermitian.
        """
        return map_fermionic_operator(
            self.to_fermionic_op(),
            mapper,
            name="electronic Hamiltonian",
        )


__all__ = ["ElectronicEnergy"]
