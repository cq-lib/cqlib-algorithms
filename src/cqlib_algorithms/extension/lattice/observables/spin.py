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

"""Spin-one-half observables for supported lattice models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from cqlib.qis import Hamiltonian

from ...many_body import SpinAxis, SpinHalfMapper, SpinOp
from ..heisenberg import HeisenbergModel
from ..ising import IsingModel

SpinLatticeModel: TypeAlias = HeisenbergModel | IsingModel


def _validate_model(model: SpinLatticeModel) -> None:
    """Validate a supported spin lattice model."""
    if not isinstance(model, (HeisenbergModel, IsingModel)):
        raise TypeError("model must be a HeisenbergModel or IsingModel")


def _validate_site(model: SpinLatticeModel, site: int, *, name: str) -> None:
    """Validate a spin-model site index."""
    if type(site) is not int:
        raise TypeError(f"{name} must be an int")
    if site < 0 or site >= model.lattice.num_nodes:
        raise ValueError(f"{name} is out of range")


def _validate_axis(axis: SpinAxis) -> None:
    """Validate a Cartesian spin axis."""
    if not isinstance(axis, str):
        raise TypeError("axis must be a str")
    if axis not in ("X", "Y", "Z"):
        raise ValueError("axis must be X, Y, or Z")


def _map_spin_observable(operator: SpinOp) -> Hamiltonian:
    return SpinHalfMapper().map_operator(operator).normalized_hermitian()


def _field_operator(model: SpinLatticeModel) -> SpinOp:
    if isinstance(model, HeisenbergModel):
        axes: tuple[SpinAxis, SpinAxis, SpinAxis] = ("X", "Y", "Z")
        return SpinOp.from_terms(
            model.lattice.num_nodes,
            (
                (((axis, site),), 2.0 * field)
                for site, fields in enumerate(model.external_fields)
                for axis, field in zip(axes, fields, strict=True)
                if field != 0.0
            ),
        )
    return SpinOp.from_terms(
        model.lattice.num_nodes,
        (
            ((("X", site),), 2.0 * field)
            for site, field in enumerate(model.transverse_fields)
            if field != 0.0
        ),
    )


@dataclass(frozen=True, slots=True)
class TotalMagnetization:
    r"""Factory for physical :math:`M_z=\sum_i S_i^z=\frac12\sum_i Z_i`.

    Args:
        model: Supported spin-one-half lattice model.

    Raises:
        TypeError: If ``model`` has the wrong type.
    """

    model: SpinLatticeModel

    def __post_init__(self) -> None:
        _validate_model(self.model)

    def to_spin_op(self) -> SpinOp:
        """Return total physical magnetization along z."""
        return SpinOp.from_terms(
            self.model.lattice.num_nodes,
            (((("Z", site),), 1.0) for site in range(self.model.lattice.num_nodes)),
        )

    def to_qubit_hamiltonian(self) -> Hamiltonian:
        """Return the exact spin-one-half qubit observable."""
        return _map_spin_observable(self.to_spin_op())


@dataclass(frozen=True, slots=True)
class SiteMagnetization:
    r"""Factory for physical :math:`M_{z,i}=S_i^z=Z_i/2`.

    Args:
        model: Supported spin-one-half lattice model.
        site: In-range lattice site.

    Raises:
        TypeError: If the model or site has the wrong type.
        ValueError: If ``site`` is outside the lattice.
    """

    model: SpinLatticeModel
    site: int

    def __post_init__(self) -> None:
        _validate_model(self.model)
        _validate_site(self.model, self.site, name="site")

    def to_spin_op(self) -> SpinOp:
        """Return physical magnetization at the selected site."""
        return SpinOp.from_terms(
            self.model.lattice.num_nodes,
            (((("Z", self.site),), 1.0),),
        )

    def to_qubit_hamiltonian(self) -> Hamiltonian:
        """Return the exact spin-one-half qubit observable."""
        return _map_spin_observable(self.to_spin_op())


@dataclass(frozen=True, slots=True)
class SpinCorrelation:
    r"""Factory for the unconnected product :math:`\sigma_i^a\sigma_j^a`.

    Equal sites are allowed and yield the identity because
    :math:`(\sigma_i^a)^2=I`. This component does not perform connected-
    correlator post-processing; callers may separately subtract
    :math:`\langle\sigma_i^a\rangle\langle\sigma_j^a\rangle` from measured
    expectations.

    Args:
        model: Supported spin-one-half lattice model.
        source: In-range first site.
        target: In-range second site; equal sites are allowed.
        axis: Cartesian Pauli axis.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a site or axis is invalid.
    """

    model: SpinLatticeModel
    source: int
    target: int
    axis: SpinAxis

    def __post_init__(self) -> None:
        _validate_model(self.model)
        _validate_site(self.model, self.source, name="source")
        _validate_site(self.model, self.target, name="target")
        _validate_axis(self.axis)

    def to_spin_op(self) -> SpinOp:
        r"""Return :math:`4S_i^aS_j^a`, equal to the requested Pauli product."""
        return SpinOp.from_terms(
            self.model.lattice.num_nodes,
            ((((self.axis, self.source), (self.axis, self.target)), 4.0),),
        )

    def to_qubit_hamiltonian(self) -> Hamiltonian:
        """Return the exact spin-one-half qubit observable."""
        return _map_spin_observable(self.to_spin_op())


@dataclass(frozen=True, slots=True)
class SpinInteractionEnergy:
    """Factory for the model interaction energy with local fields removed.

    Args:
        model: Supported spin-one-half lattice model.

    Raises:
        TypeError: If ``model`` has the wrong type.
    """

    model: SpinLatticeModel

    def __post_init__(self) -> None:
        _validate_model(self.model)

    def to_spin_op(self) -> SpinOp:
        """Return all pair-interaction terms and no local-field terms."""
        return self.model.spin_operator() - _field_operator(self.model)

    def to_qubit_hamiltonian(self) -> Hamiltonian:
        """Return the exact spin-one-half interaction Hamiltonian."""
        return _map_spin_observable(self.to_spin_op())


__all__ = [
    "SiteMagnetization",
    "SpinCorrelation",
    "SpinInteractionEnergy",
    "TotalMagnetization",
]
