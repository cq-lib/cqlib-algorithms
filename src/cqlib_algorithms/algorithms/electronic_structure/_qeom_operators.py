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

"""Exact fermionic operator algebra used by neutral electronic qEOM."""

from __future__ import annotations

from collections.abc import Sequence

from cqlib.qis import FermionicLadder, FermionicOp

from ...extension.chemistry.fermionic import FermionicExcitation


def _from_weighted_operators(
    num_modes: int,
    operators: Sequence[tuple[complex, FermionicOp]],
) -> FermionicOp:
    terms: list[tuple[tuple[FermionicLadder, ...], complex]] = []
    for factor, operator in operators:
        if operator.num_modes != num_modes:
            raise ValueError("fermionic operators must have the same mode count")
        terms.extend(
            (term, factor * coefficient) for term, coefficient in operator.terms
        )
    result = FermionicOp.from_terms(num_modes, terms).normal_order()
    result.chop(atol=0.0)
    return result


def _product(left: FermionicOp, right: FermionicOp) -> FermionicOp:
    if not isinstance(left, FermionicOp) or not isinstance(right, FermionicOp):
        raise TypeError("fermionic products require FermionicOp operands")
    if left.num_modes != right.num_modes:
        raise ValueError("fermionic product operands must have the same mode count")
    result = (left * right).normal_order()
    result.chop(atol=0.0)
    return result


def _commutator(left: FermionicOp, right: FermionicOp) -> FermionicOp:
    """Return the exact ordinary commutator ``left*right-right*left``."""
    return _from_weighted_operators(
        left.num_modes,
        ((1.0, _product(left, right)), (-1.0, _product(right, left))),
    )


def _symmetrized_double_commutator(
    left: FermionicOp,
    middle: FermionicOp,
    right: FermionicOp,
) -> FermionicOp:
    """Return ``1/2 ([[left,middle],right]+[left,[middle,right]])``."""
    if left.num_modes != middle.num_modes or middle.num_modes != right.num_modes:
        raise ValueError("fermionic operators must have the same mode count")
    return _from_weighted_operators(
        left.num_modes,
        (
            (1.0, _product(_product(left, middle), right)),
            (1.0, _product(_product(right, middle), left)),
            (-0.5, _product(_product(middle, left), right)),
            (-0.5, _product(_product(right, left), middle)),
            (-0.5, _product(_product(left, right), middle)),
            (-0.5, _product(_product(middle, right), left)),
        ),
    )


def _hermitian_components(
    operator: FermionicOp,
) -> tuple[FermionicOp | None, FermionicOp | None]:
    """Return nonzero Hermitian real and imaginary operator components."""
    adjoint = operator.adjoint()
    real_part = _from_weighted_operators(
        operator.num_modes,
        ((0.5, operator), (0.5, adjoint)),
    )
    imaginary_part = _from_weighted_operators(
        operator.num_modes,
        ((-0.5j, operator), (0.5j, adjoint)),
    )
    return (
        None if real_part.is_zero() else real_part,
        None if imaginary_part.is_zero() else imaginary_part,
    )


def _qeom_operator_matrices(
    hamiltonian: FermionicOp,
    excitations: Sequence[FermionicExcitation],
) -> tuple[
    tuple[tuple[FermionicOp, ...], ...],
    tuple[tuple[FermionicOp, ...], ...],
    tuple[tuple[FermionicOp, ...], ...],
    tuple[tuple[FermionicOp, ...], ...],
]:
    """Build exact fermionic ``M, Q, V, W`` operator matrices."""
    if not isinstance(hamiltonian, FermionicOp):
        raise TypeError("hamiltonian must be a FermionicOp")
    pool = tuple(excitations)
    if not pool:
        raise ValueError("excitations must not be empty")
    if any(not isinstance(item, FermionicExcitation) for item in pool):
        raise TypeError("excitations must contain FermionicExcitation values")
    if any(item.num_spin_orbitals != hamiltonian.num_modes for item in pool):
        raise ValueError("excitations and Hamiltonian must have the same mode count")

    excitation_ops = tuple(item.to_fermionic_op() for item in pool)
    adjoints = tuple(item.adjoint() for item in excitation_ops)
    m_rows: list[tuple[FermionicOp, ...]] = []
    q_rows: list[tuple[FermionicOp, ...]] = []
    v_rows: list[tuple[FermionicOp, ...]] = []
    w_rows: list[tuple[FermionicOp, ...]] = []
    for left in adjoints:
        m_rows.append(
            tuple(
                _symmetrized_double_commutator(left, hamiltonian, right)
                for right in excitation_ops
            )
        )
        q_rows.append(
            tuple(
                _symmetrized_double_commutator(left, hamiltonian, right).scaled(-1.0)
                for right in adjoints
            )
        )
        v_rows.append(tuple(_commutator(left, right) for right in excitation_ops))
        w_rows.append(
            tuple(_commutator(left, right).scaled(-1.0) for right in adjoints)
        )
    return tuple(m_rows), tuple(q_rows), tuple(v_rows), tuple(w_rows)


__all__: list[str] = []
