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

"""Stable first-fit grouping of Pauli terms by local measurement basis.

References
----------
[1] V. Verteletskyi, T.-C. Yen, and A. F. Izmaylov, "Measurement optimization
    in the variational quantum eigensolver using a minimum clique cover,"
    Journal of Chemical Physics, 152, 124114, 2020.
    DOI: 10.1063/1.5141458.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.qis import Pauli, PauliString


def snapshot_terms(terms: Sequence[PauliString]) -> tuple[PauliString, ...]:
    """Copy an ordered, equal-width Pauli sequence, including its phases."""
    if not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of PauliString values")
    if any(not isinstance(term, PauliString) for term in terms):
        raise TypeError("terms must contain only PauliString values")
    snapshots = tuple(term.copy() for term in terms)
    if snapshots and any(
        term.num_qubits != snapshots[0].num_qubits for term in snapshots
    ):
        raise ValueError("all Pauli terms must have the same qubit count")
    return snapshots


def qwc_compatible(left: PauliString, right: PauliString) -> bool:
    """Test qubit-wise commutation, ignoring scalar Pauli phases."""
    identity = Pauli.i()
    return all((a == identity or b == identity or a == b) for a, b in zip(left, right))


@final
class PauliGroupingStrategy:
    """Group ordered Pauli terms with ``none`` or qubit-wise commuting rules.

    ``none`` places each term in its own group. ``qwc`` uses stable first-fit:
    each term joins the earliest existing group compatible with every member,
    or starts a new group. The result contains original input indices in
    stable order. Scalar phases do not affect basis compatibility. Empty input
    yields an empty tuple; mixed widths and non-Pauli values are rejected.
    This algebraic strategy does not allocate circuits or interpret weights.
    """

    __slots__ = ("_mode",)

    def __init__(self, mode: str = "qwc") -> None:
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode not in ("none", "qwc"):
            raise ValueError("mode must be 'none' or 'qwc'")
        self._mode = mode

    @property
    def mode(self) -> str:
        """Return the selected grouping rule."""
        return self._mode

    def group(self, terms: Sequence[PauliString]) -> tuple[tuple[int, ...], ...]:
        """Return stable groups of zero-based positions in ``terms``."""
        snapshots = snapshot_terms(terms)
        if self._mode == "none":
            return tuple((index,) for index in range(len(snapshots)))

        groups: list[list[int]] = []
        for index, term in enumerate(snapshots):
            for group in groups:
                if all(qwc_compatible(term, snapshots[other]) for other in group):
                    group.append(index)
                    break
            else:
                groups.append([index])
        return tuple(tuple(group) for group in groups)
