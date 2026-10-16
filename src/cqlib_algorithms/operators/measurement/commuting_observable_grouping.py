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

"""Algebraic grouping of globally commuting Pauli observables.

References
----------
[1] O. Crawford et al., "Efficient quantum measurement of Pauli operators in
    the presence of finite sampling error," Quantum, 5, 385, 2021.
    DOI: 10.22331/q-2021-01-20-385.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from cqlib.qis import PauliString

from .pauli_grouping_strategy import snapshot_terms


@final
class CommutingObservableGrouping:
    """Partition Pauli terms by full symplectic commutation, using first-fit.

    Results are stable groups of original term indices. For example, XX and
    YY commute globally but cannot be read in a single product-Pauli basis.
    This component supplies only the algebraic partition; it must not be used
    to infer local measurement circuits from its groups. Empty input yields
    no groups. Mixed widths and non-Pauli values are rejected.
    """

    __slots__ = ()

    def group(self, terms: Sequence[PauliString]) -> tuple[tuple[int, ...], ...]:
        """Return stable first-fit globally commuting index groups."""
        snapshots = snapshot_terms(terms)
        groups: list[list[int]] = []
        for index, term in enumerate(snapshots):
            for group in groups:
                if all(term.commutes_with(snapshots[other]) for other in group):
                    group.append(index)
                    break
            else:
                groups.append([index])
        return tuple(tuple(group) for group in groups)
