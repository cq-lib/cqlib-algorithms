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

"""Spinful Fermi--Hubbard lattice model."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Literal

from cqlib.qis import (
    FermionicLadder,
    FermionicOp,
    FermionToQubitMapper,
    Hamiltonian,
    JordanWignerMapper,
)

from ._fermionic_layout import FERMION_SPINS, fermionic_mode
from ._model_validation import RealVectorParameter, real_vector, require_lattice
from .topology import Lattice


@dataclass(frozen=True, slots=True)
class FermiHubbardModel:
    r"""Spinful Fermi--Hubbard model with interleaved spin-orbital modes.

    For site ``i``, mode ``2*i`` is spin up and mode ``2*i + 1`` is spin
    down. An oriented non-self lattice edge ``(i, j, t)`` contributes

    .. math::

        t c^\dagger_{i\sigma} c_{j\sigma}
        + t^* c^\dagger_{j\sigma} c_{i\sigma}

    for each spin. A self-loop weight is the site's spin-independent onsite
    potential, while ``onsite_interaction`` supplies :math:`U_i n_{i\uparrow}
    n_{i\downarrow}`. No implicit minus sign is applied to edge weights.

    Args:
        lattice: Weighted hopping graph and onsite potentials.
        onsite_interaction: Uniform or site-dependent real Hubbard interaction.

    Raises:
        TypeError: If the lattice or interaction values have the wrong type.
        ValueError: If interaction lengths or values are invalid.
    """

    lattice: Lattice
    onsite_interaction: InitVar[RealVectorParameter] = 0.0
    _onsite_interactions: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self, onsite_interaction: RealVectorParameter) -> None:
        require_lattice(self.lattice)
        lattice = self.lattice
        interactions = real_vector(
            onsite_interaction,
            length=lattice.num_nodes,
            name="onsite_interaction",
        )
        object.__setattr__(self, "_onsite_interactions", interactions)

    @property
    def num_modes(self) -> int:
        """Return two spin-orbital modes per lattice node."""
        return 2 * self.lattice.num_nodes

    @property
    def onsite_interactions(self) -> tuple[float, ...]:
        """Return the immutable onsite interaction vector."""
        return self._onsite_interactions

    def mode_index(self, site: int, spin: Literal["up", "down"]) -> int:
        """Return the interleaved spin-orbital mode for one model site."""
        if type(site) is not int:
            raise TypeError("site must be an int")
        if site < 0 or site >= self.lattice.num_nodes:
            raise ValueError("site is out of range")
        return fermionic_mode(site, spin)

    def fermionic_operator(self) -> FermionicOp:
        """Return the second-quantized many-body Hamiltonian."""
        terms: list[tuple[tuple[FermionicLadder, ...], complex]] = []
        for source, target, weight in self.lattice.weighted_edges:
            if source == target:
                if weight != 0.0:
                    for spin in FERMION_SPINS:
                        mode = fermionic_mode(source, spin)
                        terms.append(
                            (
                                (
                                    FermionicLadder.create(mode),
                                    FermionicLadder.annihilate(mode),
                                ),
                                weight,
                            )
                        )
                continue
            for spin in FERMION_SPINS:
                source_mode = fermionic_mode(source, spin)
                target_mode = fermionic_mode(target, spin)
                terms.extend(
                    (
                        (
                            (
                                FermionicLadder.create(source_mode),
                                FermionicLadder.annihilate(target_mode),
                            ),
                            weight,
                        ),
                        (
                            (
                                FermionicLadder.create(target_mode),
                                FermionicLadder.annihilate(source_mode),
                            ),
                            weight.conjugate(),
                        ),
                    )
                )

        for node, interaction in enumerate(self._onsite_interactions):
            if interaction == 0.0:
                continue
            up = fermionic_mode(node, "up")
            down = fermionic_mode(node, "down")
            terms.append(
                (
                    (
                        FermionicLadder.create(up),
                        FermionicLadder.annihilate(up),
                        FermionicLadder.create(down),
                        FermionicLadder.annihilate(down),
                    ),
                    complex(interaction),
                )
            )
        return FermionicOp.from_terms(self.num_modes, terms)

    def total_particle_number(self) -> FermionicOp:
        """Return the total-particle-number observable."""
        return FermionicOp.total_number(self.num_modes)

    def up_particle_number(self) -> FermionicOp:
        """Return the spin-up particle-number observable."""
        return FermionicOp.from_terms(
            self.num_modes,
            (
                (
                    (
                        FermionicLadder.create(fermionic_mode(node, "up")),
                        FermionicLadder.annihilate(fermionic_mode(node, "up")),
                    ),
                    1.0,
                )
                for node in range(self.lattice.num_nodes)
            ),
        )

    def down_particle_number(self) -> FermionicOp:
        """Return the spin-down particle-number observable."""
        return FermionicOp.from_terms(
            self.num_modes,
            (
                (
                    (
                        FermionicLadder.create(fermionic_mode(node, "down")),
                        FermionicLadder.annihilate(fermionic_mode(node, "down")),
                    ),
                    1.0,
                )
                for node in range(self.lattice.num_nodes)
            ),
        )

    def qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the fermionic model to a normalized Hermitian Hamiltonian."""
        selected: FermionToQubitMapper = (
            JordanWignerMapper() if mapper is None else mapper
        )
        if not isinstance(selected, FermionToQubitMapper):
            raise TypeError("mapper must implement FermionToQubitMapper")
        mapped = selected.map_operator(self.fermionic_operator())
        if not isinstance(mapped, Hamiltonian):
            raise TypeError("mapper.map_operator() must return a Hamiltonian")
        expected_width = selected.num_qubits(self.num_modes)
        if mapped.num_qubits != expected_width:
            raise ValueError("mapper returned a Hamiltonian with an inconsistent width")
        return mapped.normalized_hermitian()
