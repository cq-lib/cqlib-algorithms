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

"""Fermionic observables for spinful lattice models."""

from __future__ import annotations

from dataclasses import dataclass

from cqlib.qis import FermionicOp, FermionToQubitMapper, Hamiltonian

from ...many_body._fermionic_mapping import map_fermionic_operator
from ..fermi_hubbard import FermiHubbardModel


def _validate_model(model: FermiHubbardModel) -> None:
    """Validate a Fermi--Hubbard model."""
    if not isinstance(model, FermiHubbardModel):
        raise TypeError("model must be a FermiHubbardModel")


def _validate_site(model: FermiHubbardModel, site: int) -> None:
    """Validate a model site index."""
    if type(site) is not int:
        raise TypeError("site must be an int")
    if site < 0 or site >= model.lattice.num_nodes:
        raise ValueError("site is out of range")


def _map_observable(
    operator: FermionicOp,
    mapper: FermionToQubitMapper | None,
    *,
    name: str,
) -> Hamiltonian:
    return map_fermionic_operator(operator, mapper, name=name)


@dataclass(frozen=True, slots=True)
class TotalParticleNumber:
    r"""Factory for :math:`N=\sum_i(n_{i\uparrow}+n_{i\downarrow})`.

    Args:
        model: Spinful Fermi--Hubbard model.

    Raises:
        TypeError: If ``model`` has the wrong type.
    """

    model: FermiHubbardModel

    def __post_init__(self) -> None:
        _validate_model(self.model)

    def to_fermionic_op(self) -> FermionicOp:
        """Return the model's total-particle-number operator."""
        return self.model.total_particle_number()

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the observable through the selected fermion mapper."""
        return _map_observable(
            self.to_fermionic_op(),
            mapper,
            name="total-particle-number observable",
        )


@dataclass(frozen=True, slots=True)
class SpinUpParticleNumber:
    r"""Factory for :math:`N_\uparrow=\sum_i n_{i\uparrow}`.

    Args:
        model: Spinful Fermi--Hubbard model.

    Raises:
        TypeError: If ``model`` has the wrong type.
    """

    model: FermiHubbardModel

    def __post_init__(self) -> None:
        _validate_model(self.model)

    def to_fermionic_op(self) -> FermionicOp:
        """Return the model's spin-up particle-number operator."""
        return self.model.up_particle_number()

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the observable through the selected fermion mapper."""
        return _map_observable(
            self.to_fermionic_op(),
            mapper,
            name="spin-up particle-number observable",
        )


@dataclass(frozen=True, slots=True)
class SpinDownParticleNumber:
    r"""Factory for :math:`N_\downarrow=\sum_i n_{i\downarrow}`.

    Args:
        model: Spinful Fermi--Hubbard model.

    Raises:
        TypeError: If ``model`` has the wrong type.
    """

    model: FermiHubbardModel

    def __post_init__(self) -> None:
        _validate_model(self.model)

    def to_fermionic_op(self) -> FermionicOp:
        """Return the model's spin-down particle-number operator."""
        return self.model.down_particle_number()

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the observable through the selected fermion mapper."""
        return _map_observable(
            self.to_fermionic_op(),
            mapper,
            name="spin-down particle-number observable",
        )


@dataclass(frozen=True, slots=True)
class SiteOccupation:
    r"""Factory for :math:`n_i=n_{i\uparrow}+n_{i\downarrow}`.

    Args:
        model: Spinful Fermi--Hubbard model.
        site: In-range lattice site.

    Raises:
        TypeError: If the model or site has the wrong type.
        ValueError: If ``site`` is outside the lattice.
    """

    model: FermiHubbardModel
    site: int

    def __post_init__(self) -> None:
        _validate_model(self.model)
        _validate_site(self.model, self.site)

    def to_fermionic_op(self) -> FermionicOp:
        """Return the total occupation of the selected site."""
        up_mode = 2 * self.site
        down_mode = up_mode + 1
        return FermionicOp.number(
            self.model.num_modes,
            up_mode,
        ) + FermionicOp.number(self.model.num_modes, down_mode)

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the observable through the selected fermion mapper."""
        return _map_observable(
            self.to_fermionic_op(),
            mapper,
            name="site-occupation observable",
        )


@dataclass(frozen=True, slots=True)
class DoubleOccupation:
    r"""Factory for :math:`D_i=n_{i\uparrow}n_{i\downarrow}`.

    Args:
        model: Spinful Fermi--Hubbard model.
        site: In-range lattice site.

    Raises:
        TypeError: If the model or site has the wrong type.
        ValueError: If ``site`` is outside the lattice.
    """

    model: FermiHubbardModel
    site: int

    def __post_init__(self) -> None:
        _validate_model(self.model)
        _validate_site(self.model, self.site)

    def to_fermionic_op(self) -> FermionicOp:
        """Return the double-occupation projector of the selected site."""
        up_mode = 2 * self.site
        down_mode = up_mode + 1
        return FermionicOp.number(
            self.model.num_modes,
            up_mode,
        ) * FermionicOp.number(self.model.num_modes, down_mode)

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the observable through the selected fermion mapper."""
        return _map_observable(
            self.to_fermionic_op(),
            mapper,
            name="double-occupation observable",
        )


@dataclass(frozen=True, slots=True)
class FermionicMagnetization:
    r"""Factory for :math:`S_z=(N_\uparrow-N_\downarrow)/2`.

    Args:
        model: Spinful Fermi--Hubbard model.

    Raises:
        TypeError: If ``model`` has the wrong type.
    """

    model: FermiHubbardModel

    def __post_init__(self) -> None:
        _validate_model(self.model)

    def to_fermionic_op(self) -> FermionicOp:
        """Return the physical total spin projection along z."""
        return self.model.up_particle_number().scaled(
            0.5
        ) + self.model.down_particle_number().scaled(-0.5)

    def to_qubit_hamiltonian(
        self,
        mapper: FermionToQubitMapper | None = None,
    ) -> Hamiltonian:
        """Map the observable through the selected fermion mapper."""
        return _map_observable(
            self.to_fermionic_op(),
            mapper,
            name="fermionic-magnetization observable",
        )


__all__ = [
    "DoubleOccupation",
    "FermionicMagnetization",
    "SiteOccupation",
    "SpinDownParticleNumber",
    "SpinUpParticleNumber",
    "TotalParticleNumber",
]
