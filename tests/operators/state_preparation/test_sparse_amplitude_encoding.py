"""Tests for exact sparse-support amplitude encoding."""

import cmath
import math

import numpy as np
import pytest

from cqlib.circuit import Circuit
from cqlib_algorithms.operators.state_preparation import SparseAmplitudeEncoding
from cqlib_algorithms.operators.state_preparation._state_synthesis import (
    _append_two_level_allocation,
)


@pytest.mark.reference
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5, 6])
def test_random_sparse_states_match_every_amplitude(num_qubits: int) -> None:
    rng = np.random.default_rng(8200 + num_qubits)
    support_size = min(5, 1 << num_qubits)
    for _ in range(4):
        indices = np.sort(rng.choice(1 << num_qubits, size=support_size, replace=False))
        values = rng.normal(size=support_size) + 1j * rng.normal(size=support_size)
        values /= np.linalg.norm(values)
        shuffled = rng.permutation(support_size)
        entries = [(int(indices[i]), complex(values[i])) for i in shuffled]
        preparation = SparseAmplitudeEncoding(num_qubits, entries)
        actual = preparation.build_circuit("unused").to_matrix()[:, 0]
        expected = np.zeros(1 << num_qubits, dtype=complex)
        expected[indices] = values
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_two_level_givens_changes_only_its_endpoint_subspace() -> None:
    num_qubits = 3
    theta = 1.173
    phase = -0.421
    for reference in range(1 << num_qubits):
        for destination in range(1 << num_qubits):
            if reference == destination:
                continue
            circuit = Circuit(num_qubits)
            _append_two_level_allocation(
                circuit,
                num_qubits=num_qubits,
                reference=reference,
                destination=destination,
                theta=theta,
                phase=phase,
            )
            matrix = circuit.to_matrix()
            outside = [
                index
                for index in range(1 << num_qubits)
                if index not in (reference, destination)
            ]
            np.testing.assert_allclose(
                matrix[np.ix_(outside, outside)],
                np.eye(len(outside)),
                rtol=0.0,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                matrix[np.ix_(outside, [reference, destination])],
                0.0,
                rtol=0.0,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                matrix[np.ix_([reference, destination], outside)],
                0.0,
                rtol=0.0,
                atol=1e-13,
            )
            expected_column = np.zeros(1 << num_qubits, dtype=complex)
            expected_column[reference] = math.cos(theta / 2)
            expected_column[destination] = cmath.exp(1j * phase) * math.sin(theta / 2)
            np.testing.assert_allclose(
                matrix[:, reference], expected_column, rtol=0.0, atol=1e-13
            )


def test_normalizes_copies_and_preserves_global_phase() -> None:
    entries = [(7, 2j), (1, -1.0)]
    preparation = SparseAmplitudeEncoding(3, entries)
    entries[:] = [(0, 1.0)]
    expected = np.zeros(8, dtype=complex)
    expected[1] = -1 / np.sqrt(5)
    expected[7] = 2j / np.sqrt(5)
    np.testing.assert_allclose(
        preparation.build_circuit("").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-13,
    )
    preparation.validate()


def test_preserves_representable_extremely_small_sparse_amplitude() -> None:
    tiny = 1e-200 * np.exp(-0.731j)
    preparation = SparseAmplitudeEncoding(3, [(1, tiny), (6, 1.0)])
    actual = preparation.build_circuit("").to_matrix()[:, 0]
    np.testing.assert_allclose(actual[6], 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual[1], tiny, rtol=1e-12, atol=0.0)
    outside = np.delete(actual, [1, 6])
    np.testing.assert_allclose(outside, 0.0, rtol=0.0, atol=1e-14)


def test_normalizes_finite_complex_entries_whose_unscaled_modulus_overflows() -> None:
    value = complex(1.3e308, 1.3e308)
    preparation = SparseAmplitudeEncoding(1, [(0, value), (1, value.conjugate())])
    expected = np.array([0.5 + 0.5j, 0.5 - 0.5j])
    np.testing.assert_allclose(
        preparation.build_circuit("").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-13,
    )


def test_large_width_build_does_not_expand_the_state_space() -> None:
    preparation = SparseAmplitudeEncoding(64, [(0, 1.0), ((1 << 64) - 1, 1j)])
    circuit = preparation.build_circuit("unused")
    assert circuit.num_qubits == 64
    assert len(circuit.operations) < 20_000


@pytest.mark.parametrize(
    "num_qubits,entries,error",
    [
        (True, [(0, 1.0)], TypeError),
        (0, [(0, 1.0)], ValueError),
        (2, [], ValueError),
        (2, [(True, 1.0)], TypeError),
        (2, [(-1, 1.0)], ValueError),
        (2, [(4, 1.0)], ValueError),
        (2, [(1, 1.0), (1, 2.0)], ValueError),
        (2, [(1, 0.0)], ValueError),
        (2, [(1, float("inf"))], ValueError),
        (2, [(1, object())], TypeError),
        (2, [(1, 1.0, 2.0)], TypeError),
        (2, "bad", TypeError),
    ],
)
def test_rejects_invalid_sparse_inputs(
    num_qubits: object, entries: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        SparseAmplitudeEncoding(num_qubits, entries)  # type: ignore[arg-type]
