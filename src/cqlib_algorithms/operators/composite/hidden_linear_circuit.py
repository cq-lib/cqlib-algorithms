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

"""Hidden-linear-function query-circuit construction.

References
----------
[1] S. Bravyi, D. Gosset, and R. Koenig, "Quantum advantage with shallow
    circuits," Science, 362(6412), 308-311, 2018.
    arXiv:1704.00690.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import numpy.typing as npt
from cqlib import Circuit

_BinaryMatrix = tuple[tuple[int, ...], ...]


def _canonicalize_adjacency_matrix(
    adjacency_matrix: npt.ArrayLike,
) -> _BinaryMatrix:
    if isinstance(adjacency_matrix, (str, bytes)):
        raise TypeError("adjacency_matrix must be a two-dimensional array-like")

    if isinstance(adjacency_matrix, np.ndarray):
        if adjacency_matrix.ndim != 2:
            raise ValueError("adjacency_matrix must be two-dimensional")
        rows: object = adjacency_matrix.tolist()
    else:
        rows = adjacency_matrix

    if not isinstance(rows, Sequence):
        raise TypeError("adjacency_matrix must be a two-dimensional array-like")
    if not rows:
        raise ValueError("adjacency_matrix must not be empty")

    size = len(rows)
    canonical_rows: list[tuple[int, ...]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError("adjacency_matrix rows must be sequences")
        if len(row) != size:
            raise ValueError("adjacency_matrix must be square")

        canonical_row: list[int] = []
        for entry in row:
            if not isinstance(entry, Integral):
                raise TypeError("adjacency_matrix entries must be integers")
            value = int(entry)
            if value not in (0, 1):
                raise ValueError("adjacency_matrix entries must be binary")
            canonical_row.append(value)
        canonical_rows.append(tuple(canonical_row))

    canonical = tuple(canonical_rows)
    for row in range(size):
        for column in range(row + 1, size):
            if canonical[row][column] != canonical[column][row]:
                raise ValueError("adjacency_matrix must be symmetric")
    return canonical


class HiddenLinearFunctionCircuit:
    r"""Construct the query circuit for a hidden linear function instance.

    For a symmetric binary matrix ``A``, define

    ``q_A(x) = sum_i A[i,i] x_i + 2 sum_(i<j) A[i,j] x_i x_j (mod 4)``.

    The returned circuit implements ``H**n D_A H**n``, where
    ``D_A|x> = i**q_A(x)|x>``. Diagonal matrix entries produce ``S`` gates and
    off-diagonal entries produce ``CZ`` gates. Qubit ``i`` corresponds to row
    and column ``i`` and is bit ``i`` in Cqlib's little-endian basis index.

    Measuring the output from ``|0...0>`` yields a vector ``z`` satisfying
    ``q_A(x) = 2 z.x (mod 4)`` for every ``x`` in the binary nullspace of
    ``A``. The component builds the unmeasured circuit only; sampling and
    classical interpretation belong to an algorithm layer.

    Args:
        adjacency_matrix: Non-empty symmetric square array-like with binary
            integer entries. The input is copied into an immutable snapshot.

    Raises:
        TypeError: If the input, a row, or an entry has the wrong type.
        ValueError: If the matrix is empty, not square, non-binary, or not
            symmetric.
    """

    __slots__ = ("_adjacency_matrix",)

    def __init__(self, adjacency_matrix: npt.ArrayLike) -> None:
        self._adjacency_matrix = _canonicalize_adjacency_matrix(adjacency_matrix)

    @property
    def adjacency_matrix(self) -> _BinaryMatrix:
        """Return the immutable canonical adjacency matrix."""
        return self._adjacency_matrix

    @property
    def num_qubits(self) -> int:
        """Return the circuit width."""
        return len(self._adjacency_matrix)

    def build_circuit(self) -> Circuit:
        """Build a fresh unmeasured hidden-linear-function query circuit.

        Returns:
            A new circuit implementing ``H**n D_A H**n``.
        """
        circuit = Circuit(self.num_qubits)
        for qubit in range(self.num_qubits):
            circuit.h(qubit)

        for row in range(self.num_qubits):
            for column in range(row + 1, self.num_qubits):
                if self._adjacency_matrix[row][column]:
                    circuit.cz(row, column)
        for qubit in range(self.num_qubits):
            if self._adjacency_matrix[qubit][qubit]:
                circuit.s(qubit)

        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        return circuit
