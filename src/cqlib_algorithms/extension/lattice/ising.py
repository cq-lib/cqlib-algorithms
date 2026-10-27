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

"""Transverse-field Ising lattice model."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from cqlib.qis import Hamiltonian

from cqlib_algorithms.extension.many_body import SpinOp

from ._model_validation import RealVectorParameter, real_vector, require_lattice
from .heisenberg import HeisenbergModel
from .topology import Lattice


@dataclass(frozen=True, slots=True)
class IsingModel:
    r"""Pauli-normalized transverse-field Ising model.

    The implemented Hamiltonian is

    .. math:: H = \sum_{(i,j)} J_{ij} Z_i Z_j + \sum_i g_i X_i.

    Non-self lattice weights are the edge-dependent ``J`` values. The
    transverse field may be uniform or site dependent. No implicit minus sign
    is applied.

    Args:
        lattice: Real-weight interaction graph without nonzero self-loops.
        transverse_field: Uniform or site-dependent real transverse field.

    Raises:
        TypeError: If the lattice or field values have the wrong type.
        ValueError: If weights, field lengths, or values are invalid.
    """

    lattice: Lattice
    transverse_field: InitVar[RealVectorParameter] = 0.0
    _transverse_fields: tuple[float, ...] = field(init=False, repr=False)
    _heisenberg_model: HeisenbergModel = field(init=False, repr=False)

    def __post_init__(self, transverse_field: RealVectorParameter) -> None:
        require_lattice(self.lattice)
        lattice = self.lattice
        fields = real_vector(
            transverse_field,
            length=lattice.num_nodes,
            name="transverse_field",
        )
        delegate = HeisenbergModel(
            lattice,
            coupling=(0.0, 0.0, 1.0),
            external_field=tuple((field_value, 0.0, 0.0) for field_value in fields),
        )
        object.__setattr__(self, "_transverse_fields", fields)
        object.__setattr__(self, "_heisenberg_model", delegate)

    @property
    def transverse_fields(self) -> tuple[float, ...]:
        """Return the immutable site-dependent transverse-field vector."""
        return self._transverse_fields

    def spin_operator(self) -> SpinOp:
        """Return an equivalent physical-spin-one-half operator."""
        return self._heisenberg_model.spin_operator()

    def qubit_hamiltonian(self) -> Hamiltonian:
        """Return the exact Pauli Hamiltonian for the model."""
        return self._heisenberg_model.qubit_hamiltonian()
