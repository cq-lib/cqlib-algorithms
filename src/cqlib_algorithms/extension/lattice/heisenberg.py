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

"""Anisotropic spin-one-half Heisenberg lattice model."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from cqlib.qis import Hamiltonian

from cqlib_algorithms.extension.many_body import SpinAxis, SpinHalfMapper, SpinOp

from ._model_validation import (
    FieldParameter,
    RealParameter,
    RealTriple,
    external_fields,
    real_interaction_edges,
    real_triple,
    require_lattice,
)
from .topology import Lattice


@dataclass(frozen=True, slots=True)
class HeisenbergModel:
    r"""Pauli-normalized anisotropic Heisenberg model.

    The implemented Hamiltonian is

    .. math::

        H = \sum_{(i,j)} w_{ij}\sum_{a\in\{x,y,z\}} J_a\sigma_i^a\sigma_j^a
            + \sum_i \boldsymbol{h}_i\cdot\boldsymbol{\sigma}_i.

    Edge weights, couplings and fields are finite real numbers. Nonzero lattice
    self-loops are rejected because onsite fields are supplied explicitly.

    Args:
        lattice: Real-weight interaction graph without nonzero self-loops.
        coupling: Global ``(Jx, Jy, Jz)`` coupling strengths.
        external_field: Uniform or site-dependent ``(hx, hy, hz)`` fields.

    Raises:
        TypeError: If the lattice or numeric parameters have the wrong type.
        ValueError: If weights, vectors, or sequence lengths are invalid.
    """

    lattice: Lattice
    coupling: InitVar[tuple[RealParameter, RealParameter, RealParameter]] = (
        1.0,
        1.0,
        1.0,
    )
    external_field: InitVar[FieldParameter] = (0.0, 0.0, 0.0)
    _coupling: RealTriple = field(init=False, repr=False)
    _external_fields: tuple[RealTriple, ...] = field(init=False, repr=False)
    _interaction_edges: tuple[tuple[int, int, float], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(
        self,
        coupling: tuple[RealParameter, RealParameter, RealParameter],
        external_field: FieldParameter,
    ) -> None:
        require_lattice(self.lattice)
        lattice = self.lattice
        normalized_coupling = real_triple(coupling, name="coupling")
        fields = external_fields(external_field, num_nodes=lattice.num_nodes)
        edges = real_interaction_edges(lattice)
        object.__setattr__(self, "_coupling", normalized_coupling)
        object.__setattr__(self, "_external_fields", fields)
        object.__setattr__(self, "_interaction_edges", edges)

    @property
    def couplings(self) -> RealTriple:
        """Return the global ``(Jx, Jy, Jz)`` coupling vector."""
        return self._coupling

    @property
    def external_fields(self) -> tuple[RealTriple, ...]:
        """Return one immutable ``(hx, hy, hz)`` vector per site."""
        return self._external_fields

    def spin_operator(self) -> SpinOp:
        """Return an equivalent physical-spin-one-half operator."""
        axes: tuple[SpinAxis, SpinAxis, SpinAxis] = ("X", "Y", "Z")
        terms: list[tuple[tuple[tuple[SpinAxis, int], ...], complex]] = []
        for source, target, weight in self._interaction_edges:
            for axis, coupling in zip(axes, self._coupling, strict=True):
                coefficient = 4.0 * weight * coupling
                if coefficient != 0.0:
                    terms.append(
                        (((axis, source), (axis, target)), complex(coefficient))
                    )
        for node, field_vector in enumerate(self._external_fields):
            for axis, field_strength in zip(axes, field_vector, strict=True):
                coefficient = 2.0 * field_strength
                if coefficient != 0.0:
                    terms.append((((axis, node),), complex(coefficient)))
        return SpinOp.from_terms(self.lattice.num_nodes, terms)

    def qubit_hamiltonian(self) -> Hamiltonian:
        """Return the exact Pauli Hamiltonian for the model."""
        return (
            SpinHalfMapper().map_operator(self.spin_operator()).normalized_hermitian()
        )
