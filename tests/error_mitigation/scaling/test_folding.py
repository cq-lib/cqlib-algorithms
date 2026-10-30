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

"""Tests for the gate-folding noise-scaling primitives."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib.error_mitigation import ZNEMitigation

from cqlib import Circuit
from cqlib_algorithms.error_mitigation.scaling import (
    fold_gates_at_random,
    fold_global,
    fold_local,
)


def _body() -> Circuit:
    """Return the two-qubit entangling body used across the tests."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _parametrized_body(num_gates: int = 4) -> Circuit:
    """Return a one-qubit body carrying exactly ``num_gates`` rotations."""
    circuit = Circuit(1)
    for index in range(num_gates):
        circuit.rx(0, 0.1 * (index + 1))
    return circuit


def _measured(circuit: Circuit) -> Circuit:
    """Append one measurement per qubit and return the same circuit."""
    for qubit in range(len(circuit.qubits)):
        circuit.measure(qubit)
    return circuit


def _names(circuit: Circuit) -> list[str]:
    """Return the operation names of ``circuit``, in execution order."""
    return [operation.name for operation in circuit.operations]


def _phase(circuit: Circuit) -> complex:
    """Return the global phase of ``circuit`` as a complex scalar."""
    return complex(circuit.global_phase.evaluate({}))


def _measurement_count(circuit: Circuit) -> int:
    """Return how many measurement records ``circuit`` carries."""
    return sum(1 for operation in circuit.operations if operation.is_classical_data)


def test_fold_global_leaves_scale_one_untouched() -> None:
    body = _body()
    folded = fold_global(body, 1.0)
    assert _names(folded) == _names(body)
    assert np.allclose(body.to_matrix(), folded.to_matrix())


@pytest.mark.parametrize("scale_factor", [3.0, 5.0])
def test_fold_global_integer_scale_multiplies_the_gate_count(
    scale_factor: float,
) -> None:
    body = _body()
    folded = fold_global(body, scale_factor)
    assert len(folded.operations) == int(scale_factor) * len(body.operations)
    assert np.allclose(body.to_matrix(), folded.to_matrix())


@pytest.mark.parametrize(
    ("scale_factor", "expected_gates"),
    [(1.0, 4), (2.0, 8), (3.0, 12), (3.4, 14), (4.0, 16), (5.6, 22)],
)
def test_fold_global_realises_the_requested_scale_factor(
    scale_factor: float,
    expected_gates: int,
) -> None:
    body = _parametrized_body(4)
    folded = fold_global(body, scale_factor)
    assert len(folded.operations) == expected_gates
    # The residual is quantised to half a gate, so the realised factor lands
    # within one gate of the request instead of exactly on it.
    realised = len(folded.operations) / len(body.operations)
    assert abs(realised - scale_factor) <= 1.0 / len(body.operations)
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_global_preserves_the_global_phase() -> None:
    body = _body()
    body.set_global_phase(0.25)
    for scale_factor in (1.0, 2.0, 3.0, 3.4):
        assert _phase(fold_global(body, scale_factor)) == pytest.approx(0.25)


def test_fold_global_strips_and_restores_measurements() -> None:
    measured = _measured(_body())
    folded = fold_global(measured, 2.0)
    assert _measurement_count(measured) == 2
    assert _measurement_count(folded) == 2


def test_fold_global_agrees_with_cqlib_on_odd_integer_scales() -> None:
    folded = fold_global(_parametrized_body(3), 5.0)
    expected = ZNEMitigation(_parametrized_body(3), [2]).fold_circuits()[0]
    assert _names(folded) == _names(expected)
    assert np.allclose(folded.to_matrix(), expected.to_matrix())


def test_fold_global_rejects_a_scale_below_one() -> None:
    with pytest.raises(ValueError):
        fold_global(_body(), 0.5)


def test_fold_gates_at_random_is_deterministic_in_the_seed() -> None:
    first = fold_gates_at_random(_parametrized_body(4), 3.4, seed=7)
    second = fold_gates_at_random(_parametrized_body(4), 3.4, seed=7)
    assert _names(first) == _names(second)
    assert np.allclose(first.to_matrix(), second.to_matrix())


def test_fold_gates_at_random_keeps_the_unitary_and_never_shrinks() -> None:
    body = _parametrized_body(4)
    folded = fold_gates_at_random(body, 3.4, seed=7)
    assert len(folded.operations) >= len(body.operations)
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_gates_at_random_restores_measurements() -> None:
    measured = _measured(_body())
    folded = fold_gates_at_random(measured, 3.0, seed=1)
    assert _measurement_count(folded) == 2


def test_fold_gates_at_random_rejects_a_scale_below_one() -> None:
    with pytest.raises(ValueError):
        fold_gates_at_random(_body(), 0.5, seed=0)


def test_fold_local_folds_every_chunk_globally() -> None:
    body = _parametrized_body(4)
    folded = fold_local(body, 3.0, num_chunks=2)
    assert len(folded.operations) == 3 * len(body.operations)
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_local_preserves_the_global_phase() -> None:
    body = _body()
    body.set_global_phase(0.4)
    assert _phase(fold_local(body, 3.0, num_chunks=2)) == pytest.approx(0.4)


def test_fold_local_restores_measurements() -> None:
    measured = _measured(_body())
    folded = fold_local(measured, 3.0, num_chunks=2)
    assert _measurement_count(folded) == 2


@pytest.mark.parametrize("num_chunks", [0, -1])
def test_fold_local_rejects_a_non_positive_chunk_count(num_chunks: int) -> None:
    with pytest.raises(ValueError):
        fold_local(_body(), 3.0, num_chunks=num_chunks)


def test_fold_local_rejects_a_scale_below_one() -> None:
    with pytest.raises(ValueError):
        fold_local(_body(), 0.5, num_chunks=2)


def _signature(circuit: Circuit) -> list[tuple[str, tuple[float, ...]]]:
    """Return each operation's name and parameters, in execution order.

    Gate names alone cannot tell two foldings apart, since every gate of the
    parametrized body is an ``RX``; the parameters carry the gate identity.
    """
    return [
        (operation.name, tuple(float(value) for value in operation.params))
        for operation in circuit.operations
    ]


@pytest.mark.parametrize("scale_factor", [1.0, 3.0, 5.0])
def test_fold_global_matches_the_cqlib_path_at_odd_scale_factors(
    scale_factor: float,
) -> None:
    body = _parametrized_body(3)
    folded = fold_global(body, scale_factor)

    # Every gate is folded by the same U (U^dag U)^n pattern, so the cqlib ZNE
    # helper at fold level (scale_factor - 1) / 2 must agree gate for gate, not
    # merely up to the final unitary.
    expected = ZNEMitigation(
        _parametrized_body(3),
        [int((scale_factor - 1.0) / 2.0)],
    ).fold_circuits()[0]
    assert _names(folded) == _names(expected)
    assert np.allclose(folded.to_matrix(), expected.to_matrix())


def test_fold_gates_at_random_is_reproducible_for_a_seed() -> None:
    # The residual fold lands on one randomly chosen gate, so the seed has to
    # fix which gate that is: the same seed reproduces the same sequence of
    # gate parameters and a different seed moves the folded gate elsewhere.
    first = fold_gates_at_random(_parametrized_body(4), 3.4, seed=1)
    repeated = fold_gates_at_random(_parametrized_body(4), 3.4, seed=1)
    other = fold_gates_at_random(_parametrized_body(4), 3.4, seed=2)

    assert _signature(first) == _signature(repeated)
    assert _signature(other) != _signature(first)
