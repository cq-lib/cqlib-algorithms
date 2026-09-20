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

"""Grouping tests against independent local-Pauli compatibility rules."""

from __future__ import annotations

import pytest
from cqlib.qis import PauliString

from cqlib_algorithms.operators.measurement import (
    CommutingObservableGrouping,
    PauliGroupingStrategy,
)


def test_none_and_qwc_first_fit_are_stable() -> None:
    terms = [PauliString.from_str(label) for label in ("XI", "IZ", "YI", "XZ")]
    assert PauliGroupingStrategy("none").group(terms) == ((0,), (1,), (2,), (3,))
    assert PauliGroupingStrategy("qwc").group(terms) == ((0, 1, 3), (2,))
    assert PauliGroupingStrategy().group(terms) == ((0, 1, 3), (2,))


def test_general_commutation_is_not_qubitwise_commutation() -> None:
    terms = [PauliString.from_str(label) for label in ("XX", "YY", "ZZ")]
    assert CommutingObservableGrouping().group(terms) == ((0, 1, 2),)
    assert PauliGroupingStrategy("qwc").group(terms) == ((0,), (1,), (2,))


def test_scalar_phase_does_not_change_grouping() -> None:
    terms = [PauliString.from_str(label) for label in ("X", "-iX", "+iI")]
    assert PauliGroupingStrategy().group(terms) == ((0, 1, 2),)
    assert CommutingObservableGrouping().group(terms) == ((0, 1, 2),)


def test_empty_groups_and_input_validation() -> None:
    assert PauliGroupingStrategy().group([]) == ()
    assert CommutingObservableGrouping().group(()) == ()
    with pytest.raises(TypeError):
        PauliGroupingStrategy(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PauliGroupingStrategy("commuting")
    with pytest.raises(TypeError):
        PauliGroupingStrategy().group("X")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CommutingObservableGrouping().group(["X"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        PauliGroupingStrategy().group([PauliString(1), PauliString(2)])
    with pytest.raises(ValueError):
        CommutingObservableGrouping().group([PauliString(1), PauliString(2)])
