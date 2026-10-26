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

"""Tests for the layer-selective noise-scaling primitive."""

from __future__ import annotations

import numpy as np
import pytest

from cqlib import Circuit
from cqlib_algorithms.error_mitigation.scaling import fold_layers


def _body() -> Circuit:
    """Return the two-operation, two-qubit body used across the tests."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _wide_body() -> Circuit:
    """Return a three-operation, two-qubit body for block-boundary tests."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(1)
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


def test_fold_layers_folds_every_addressed_layer() -> None:
    body = _body()
    folded = fold_layers(body, scales={0: 3.0, 1: 3.0})
    assert len(folded.operations) == 6
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_layers_leaves_unaddressed_layers_untouched() -> None:
    body = _body()
    folded = fold_layers(body, scales={0: 3.0})
    assert len(folded.operations) == 4
    assert _names(folded)[-1] == _names(body)[-1]


def test_fold_layers_consumes_the_fraction_of_a_block() -> None:
    body = _wide_body()
    folded = fold_layers(body, scales={0: 2.0, 1: 2.0})
    # Layers 0 and 1 form one block of two gates at lambda = 2, so n = 0 and
    # f = 1: the residual repeats round(1 * 2 / 2) = 1 gate as L^dagger L. The
    # block grows from two to four gates and layer 2 is copied through.
    assert len(folded.operations) == 4 + 1
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_layers_rounds_a_small_residual_down_to_no_tail() -> None:
    body = _body()
    folded = fold_layers(body, scales={0: 3.6})
    # n = 1 fold on layer 0; the 0.6 residual rounds to round(0.3) = 0 gates,
    # so no tail is appended and layer 1 is copied through untouched.
    assert len(folded.operations) == 4
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_layers_delimits_blocks_at_unaddressed_layers() -> None:
    body = _wide_body()
    folded = fold_layers(body, scales={0: 3.0, 2: 3.0})
    # Layers 0 and 2 are separated by the unaddressed layer 1, so each is a
    # block of one gate folded once: 3 + 1 + 3.
    assert len(folded.operations) == 7
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_layers_folds_a_layer_twice_at_scale_five() -> None:
    body = _body()
    folded = fold_layers(body, scales={1: 5.0})
    # floor((5 - 1) / 2) = 2 folds on layer 1, so it contributes 2 * 2 + 1.
    assert len(folded.operations) == 1 + 5
    assert np.allclose(body.to_matrix(), folded.to_matrix())


def test_fold_layers_at_scale_one_is_the_identity_map() -> None:
    body = _body()
    folded = fold_layers(body, scales={0: 1.0, 1: 1.0})
    assert _names(folded) == _names(body)


def test_fold_layers_with_an_empty_scale_map_is_the_identity_map() -> None:
    body = _body()
    assert _names(fold_layers(body, scales={})) == _names(body)


def test_fold_layers_preserves_the_global_phase() -> None:
    body = _body()
    body.set_global_phase(0.3)
    assert _phase(fold_layers(body, scales={0: 3.0})) == pytest.approx(0.3)


def test_fold_layers_strips_and_restores_measurements() -> None:
    measured = _measured(_body())
    folded = fold_layers(measured, scales={0: 3.0})
    assert _measurement_count(measured) == 2
    assert _measurement_count(folded) == 2
    # The addressed index refers to the gate body, not to the raw operation
    # list that also carries the measurements.
    assert len(folded.operations) == 4 + 2


@pytest.mark.parametrize("index", [2, 5])
def test_fold_layers_rejects_an_out_of_range_index(index: int) -> None:
    with pytest.raises(ValueError):
        fold_layers(_body(), scales={index: 3.0})


def test_fold_layers_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError):
        fold_layers(_body(), scales={-1: 3.0})


def test_fold_layers_rejects_a_scale_below_one() -> None:
    with pytest.raises(ValueError):
        fold_layers(_body(), scales={0: 0.5})
