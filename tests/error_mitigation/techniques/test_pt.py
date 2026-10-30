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

"""Tests for Pauli twirling."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.circuit import StandardGate

from cqlib_algorithms.error_mitigation.techniques.pt import (
    PauliTwirling,
    _CNOT_TWIRLINGS,
    _TWIRLING_TABLES,
    combine_results,
    construct_circuits,
)

_ATOL = 1e-9

_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_LABELS = tuple(_PAULI_MATRICES)
# The twirled object is a four-dimensional operator (two qubits), so the Pauli
# transfer matrix is built over the 16 two-qubit Pauli operators.
_TWO_QUBIT_BASIS = [
    (left, right, np.kron(_PAULI_MATRICES[left], _PAULI_MATRICES[right]))
    for left, right in product(_PAULI_LABELS, repeat=2)
]


def _label(gate: StandardGate) -> str:
    """Return the bare Pauli label of a standard gate, e.g. ``I`` for ``I``."""
    return str(gate).rsplit(".", 1)[-1]


def _pauli_pair_instructions(
    table: Sequence[tuple[StandardGate, StandardGate, StandardGate, StandardGate]],
) -> list[tuple[str, str, str, str]]:
    """Return every row of ``table`` as four bare Pauli labels."""
    return [
        (_label(p), _label(q), _label(r), _label(s)) for p, q, r, s in table
    ]


def _allclose_up_to_global_phase(
    left: np.ndarray,
    right: np.ndarray,
    /,
) -> bool:
    """Return whether two operators agree up to a global phase."""
    overlap = np.vdot(right, left)
    if abs(overlap) < _ATOL:
        return False
    return bool(np.allclose(left, overlap / abs(overlap) * right, atol=_ATOL))


def _coherent_pair_error(angle: float, /) -> np.ndarray:
    r"""Return the coherent two-qubit error :math:`e^{i\theta Z\otimes Z}`."""
    diagonal = np.array([1.0, -1.0, -1.0, 1.0], dtype=complex)
    return np.diag(np.exp(1.0j * angle * diagonal))


def _twirled_pauli_transfer_matrix(
    pairs: Sequence[tuple[str, str]],
    error: np.ndarray,
    /,
) -> np.ndarray:
    r"""Return the Pauli transfer matrix of the twirl of ``error``.

    The twirled channel is
    :math:`\rho \mapsto \frac{1}{N}\sum_k U_k^\dagger E \rho E^\dagger U_k`
    with :math:`U_k` the pre-gate Pauli pair of row ``k``, so entry
    :math:`(a, b)` reads :math:`\frac{1}{2^n}
    \operatorname{Tr}[P_a \Lambda(P_b)]`.
    """
    matrix = np.zeros((len(_TWO_QUBIT_BASIS), len(_TWO_QUBIT_BASIS)), dtype=float)
    for row, (_, _, left) in enumerate(_TWO_QUBIT_BASIS):
        channel_image = np.zeros((4, 4), dtype=complex)
        for first, second in pairs:
            unitary = np.kron(_PAULI_MATRICES[first], _PAULI_MATRICES[second])
            channel_image += (
                unitary.conj().T @ error @ left @ error.conj().T @ unitary
            )
        channel_image /= len(pairs)
        for column, (_, _, right) in enumerate(_TWO_QUBIT_BASIS):
            matrix[row, column] = float(
                np.real(np.trace(right @ channel_image)) / 4.0
            )
    return matrix


def test_the_twirl_table_is_complete_and_consistent() -> None:
    for gate, table in _TWIRLING_TABLES.items():
        rows = _pauli_pair_instructions(table)

        # The pre-gate pairs must run over the whole two-qubit Pauli group,
        # which is what turns the average into a full twirl.
        pairs = [(p, q) for p, q, _r, _s in rows]
        assert len(rows) == 16
        assert len(set(pairs)) == 16
        assert set(pairs) == set(product(_PAULI_LABELS, repeat=2))

        # Every row must dress the gate into the same gate: this is the
        # condition kron(R, S) @ G @ kron(P, Q) = G.
        ideal = gate.matrix()
        for p, q, r, s in rows:
            dressed = (
                np.kron(_PAULI_MATRICES[r], _PAULI_MATRICES[s])
                @ ideal
                @ np.kron(_PAULI_MATRICES[p], _PAULI_MATRICES[q])
            )
            assert _allclose_up_to_global_phase(dressed, ideal)


def test_apply_preserves_the_unitary_up_to_the_inserted_paulis(
    bell_circuit: Circuit,
) -> None:
    bell_circuit.cz(1, 0)
    ideal = bell_circuit.to_matrix()
    twirling = PauliTwirling(num_randomizations=3, seed=11)

    dressed = [twirling.apply(bell_circuit), *twirling.variants(bell_circuit)]

    for variant in dressed:
        assert _allclose_up_to_global_phase(variant.to_matrix(), ideal)


def test_variants_are_independent_but_reproducible(bell_circuit: Circuit) -> None:
    bell_circuit.cz(1, 0)
    twirling = PauliTwirling(num_randomizations=3, seed=11)

    first = twirling.variants(bell_circuit)
    second = twirling.variants(bell_circuit)

    signatures = [
        [operation.name for operation in variant.operations] for variant in first
    ]
    # One seed reproduces the whole family...
    assert signatures == [
        [operation.name for operation in variant.operations] for variant in second
    ]
    # ...while the variants inside it are drawn independently.
    assert len({tuple(signature) for signature in signatures}) > 1


def test_variants_rejects_a_non_positive_num_randomizations(
    bell_circuit: Circuit,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        construct_circuits(bell_circuit, num_randomizations=0)
    with pytest.raises(ValueError, match="at least one"):
        PauliTwirling(num_randomizations=-1)
    with pytest.raises(TypeError, match="must be an integer"):
        construct_circuits(bell_circuit, num_randomizations=2.0)  # type: ignore[arg-type]


def test_combine_results_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one result"):
        combine_results([])


def test_a_coherent_error_is_shaped_into_a_pauli_channel() -> None:
    pairs = [(p, q) for p, q, _r, _s in _pauli_pair_instructions(_CNOT_TWIRLINGS)]

    matrix = _twirled_pauli_transfer_matrix(pairs, _coherent_pair_error(0.1))

    # A Pauli channel has a diagonal transfer matrix, so the coherent
    # off-diagonal weight that made the error accumulate linearly is gone.
    off_diagonal = matrix - np.diag(np.diagonal(matrix))
    assert np.abs(off_diagonal).max() < 1e-12
    # The surviving diagonal is a probability distribution: trace preserving
    # and non-negative.
    diagonal = np.diagonal(matrix)
    assert diagonal.min() > -1e-12
    assert float(diagonal.sum()) == pytest.approx(1.0, abs=1e-12)
