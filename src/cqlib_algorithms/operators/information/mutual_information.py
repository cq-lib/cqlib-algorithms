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

"""Quantum mutual information of a qubit-state bipartition.

References
----------
[1] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
    Information*, 10th Anniversary Edition, Cambridge University Press,
    2010. DOI: 10.1017/CBO9780511976667.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from numbers import Real

from cqlib.qis import metrics
from cqlib.qis.state import DensityMatrix, Statevector


def _validate_base(base: float) -> float:
    if isinstance(base, bool) or not isinstance(base, Real):
        raise TypeError("base must be a real number")
    normalized = float(base)
    if not math.isfinite(normalized):
        raise ValueError("base must be finite")
    if normalized <= 1.0:
        raise ValueError("base must be greater than one")
    return normalized


def _validate_partition(
    subsystem_a: Sequence[int],
    num_qubits: int,
) -> tuple[int, ...]:
    if isinstance(subsystem_a, (str, bytes)) or not isinstance(subsystem_a, Sequence):
        raise TypeError("subsystem_a must be a sequence of qubit indices")

    indices = tuple(subsystem_a)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("subsystem_a must contain only integer qubit indices")
    if not indices:
        raise ValueError("subsystem_a must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("subsystem_a qubit indices must be unique")
    if any(index < 0 or index >= num_qubits for index in indices):
        raise ValueError("subsystem_a qubit indices must be in range")
    if len(indices) == num_qubits:
        raise ValueError("subsystem_a must be a proper subset of the state qubits")
    return tuple(sorted(indices))


def _as_density_matrix(state: Statevector | DensityMatrix) -> DensityMatrix:
    if isinstance(state, DensityMatrix):
        return state
    if isinstance(state, Statevector):
        return DensityMatrix.from_state(state.num_qubits, state.data)
    raise TypeError("state must be a cqlib Statevector or DensityMatrix")


def mutual_information(
    state: Statevector | DensityMatrix,
    subsystem_a: Sequence[int],
    *,
    base: float = 2.0,
) -> float:
    """Return the quantum mutual information across one qubit bipartition.

    The result is ``S(rho_A) + S(rho_B) - S(rho_AB)``, where ``B`` is the
    complement of ``subsystem_a`` and ``S`` is the von Neumann entropy. Qubit
    indices are Core positions with qubit ``0`` as the least-significant
    basis-state bit; their order inside ``subsystem_a`` does not affect the
    result. A statevector is converted to a density matrix before reduction.

    Args:
        state: A Core statevector or density matrix containing at least two
            qubits.
        subsystem_a: A non-empty proper subset of distinct qubit indices.
        base: Finite logarithm base greater than one. The default reports bits.

    Returns:
        The non-negative mutual information in the selected logarithm unit.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the state cannot be bipartitioned or an argument is out
            of range.
        ArithmeticError: If Core returns non-finite entropies or a negative
            result beyond floating-point roundoff.
    """
    density_matrix = _as_density_matrix(state)
    if density_matrix.num_qubits < 2:
        raise ValueError("state must contain at least two qubits")

    normalized_base = _validate_base(base)
    partition_a = _validate_partition(subsystem_a, density_matrix.num_qubits)
    selected = set(partition_a)
    partition_b = tuple(
        index for index in range(density_matrix.num_qubits) if index not in selected
    )

    entropy_a = float(metrics.entropy(density_matrix.partial_trace(list(partition_a))))
    entropy_b = float(metrics.entropy(density_matrix.partial_trace(list(partition_b))))
    entropy_ab = float(metrics.entropy(density_matrix))
    if not all(math.isfinite(value) for value in (entropy_a, entropy_b, entropy_ab)):
        raise ArithmeticError("Core returned a non-finite von Neumann entropy")

    value_bits = entropy_a + entropy_b - entropy_ab
    scale = max(1.0, abs(entropy_a) + abs(entropy_b) + abs(entropy_ab))
    tolerance = 128.0 * sys.float_info.epsilon * scale
    if value_bits < -tolerance:
        raise ArithmeticError(
            "computed mutual information is negative beyond numerical roundoff"
        )
    value_bits = max(0.0, value_bits)
    return value_bits / math.log2(normalized_base)
