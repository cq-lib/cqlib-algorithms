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

"""Tests for the quantum Fourier transform."""

from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from cqlib import Circuit, Qubit
from cqlib.qis import Statevector

from cqlib_algorithms.operators.composite import QFT


def _fourier_matrix(num_qubits: int) -> np.ndarray:
    dimension = 2**num_qubits
    indices = np.arange(dimension)
    phases = np.exp(2j * np.pi * np.outer(indices, indices) / dimension)
    return phases / math.sqrt(dimension)


def _reverse_bits(value: int, width: int) -> int:
    result = 0
    for index in range(width):
        result |= ((value >> index) & 1) << (width - 1 - index)
    return result


def _bit_reversal_matrix(num_qubits: int) -> np.ndarray:
    dimension = 2**num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for value in range(dimension):
        matrix[_reverse_bits(value, num_qubits), value] = 1.0
    return matrix


def _approximate_qft_reference_matrix(
    num_qubits: int,
    approximation_degree: int,
    *,
    do_swaps: bool,
    inverse: bool,
) -> np.ndarray:
    """Construct an AQFT matrix directly from computational-basis phases."""
    dimension = 2**num_qubits
    normalization = 1 / math.sqrt(dimension)
    maximum_phase_distance = num_qubits - approximation_degree - 1
    unswapped = np.empty((dimension, dimension), dtype=np.complex128)

    for input_state in range(dimension):
        for output_state in range(dimension):
            phase_units = 0.0
            for output_qubit in range(num_qubits):
                output_bit = (output_state >> output_qubit) & 1
                first_input_qubit = max(
                    0,
                    output_qubit - maximum_phase_distance,
                )
                truncated_binary_fraction = sum(
                    ((input_state >> input_qubit) & 1)
                    / (2 ** (output_qubit - input_qubit))
                    for input_qubit in range(
                        first_input_qubit,
                        output_qubit + 1,
                    )
                )
                phase_units += output_bit * truncated_binary_fraction
            unswapped[output_state, input_state] = normalization * np.exp(
                1j * math.pi * phase_units
            )

    forward = _bit_reversal_matrix(num_qubits) @ unswapped if do_swaps else unswapped
    return forward.conj().T if inverse else forward


def test_qft_configuration_signature_has_keyword_only_options():
    signature = inspect.signature(QFT)

    assert (
        signature.parameters["num_qubits"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    for name in ("approximation_degree", "do_swaps", "inverse"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        QFT(3, 1)  # type: ignore[misc]


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_forward_qft_matches_positive_exponent_fourier_matrix(num_qubits):
    actual = QFT(num_qubits).build_circuit().to_matrix()

    np.testing.assert_allclose(
        actual,
        _fourier_matrix(num_qubits),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_inverse_qft_matches_fourier_adjoint(num_qubits):
    actual = QFT(num_qubits, inverse=True).build_circuit().to_matrix()
    expected = _fourier_matrix(num_qubits).conj().T

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_qft_statevector_uses_qubit_zero_as_lsb():
    num_qubits = 3
    dimension = 2**num_qubits
    basis_state = 5
    initial = np.zeros(dimension, dtype=np.complex128)
    initial[basis_state] = 1.0
    state = Statevector.from_state(num_qubits, initial)

    state.apply_circuit(QFT(num_qubits).build_circuit())

    expected = _fourier_matrix(num_qubits)[:, basis_state]
    np.testing.assert_allclose(state.data, expected, rtol=0.0, atol=1e-12)


def test_qft_without_swaps_has_bit_reversed_output():
    num_qubits = 3
    actual = QFT(num_qubits, do_swaps=False).build_circuit().to_matrix()
    expected = _bit_reversal_matrix(num_qubits) @ _fourier_matrix(num_qubits)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("approximation_degree", [1, 2])
@pytest.mark.parametrize("do_swaps", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_approximate_qft_matches_independent_basis_phase_reference(
    approximation_degree,
    do_swaps,
    inverse,
):
    num_qubits = 4
    actual = (
        QFT(
            num_qubits,
            approximation_degree=approximation_degree,
            do_swaps=do_swaps,
            inverse=inverse,
        )
        .build_circuit()
        .to_matrix()
    )
    expected = _approximate_qft_reference_matrix(
        num_qubits,
        approximation_degree,
        do_swaps=do_swaps,
        inverse=inverse,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("approximation_degree", [0, 1, 2, 3])
def test_approximation_retains_only_the_configured_distances(
    approximation_degree,
):
    num_qubits = 4
    circuit = QFT(
        num_qubits,
        approximation_degree=approximation_degree,
        do_swaps=False,
    ).build_circuit()
    phase_operations = [
        operation for operation in circuit.operations if operation.is_mcgate
    ]
    cutoff = num_qubits - approximation_degree
    expected_pairs = [
        (control, target)
        for target in range(num_qubits - 1, -1, -1)
        for control in range(target)
        if target - control < cutoff
    ]

    assert [operation.name for operation in phase_operations] == ["C1-Phase"] * len(
        expected_pairs
    )
    assert [
        tuple(qubit.index for qubit in operation.qubits)
        for operation in phase_operations
    ] == expected_pairs
    assert [operation.params[0] for operation in phase_operations] == pytest.approx(
        [math.pi / (2 ** (target - control)) for control, target in expected_pairs]
    )
    assert all(operation.name != "CRZ" for operation in circuit.operations)


def test_inverse_is_adjoint_of_same_approximate_unswapped_configuration():
    configuration = {
        "num_qubits": 4,
        "approximation_degree": 2,
        "do_swaps": False,
    }
    forward = QFT(**configuration).build_circuit().to_matrix()
    inverse = QFT(**configuration, inverse=True).build_circuit().to_matrix()

    np.testing.assert_allclose(inverse, forward.conj().T, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    (
        "num_qubits",
        "expected_h_count",
        "expected_phase_count",
        "expected_swap_count",
        "expected_total_count",
    ),
    [
        (1, 1, 0, 0, 1),
        (2, 2, 1, 1, 4),
        (3, 3, 3, 1, 7),
        (4, 4, 6, 2, 12),
    ],
)
def test_exact_qft_contains_only_expected_gates_and_validates(
    num_qubits,
    expected_h_count,
    expected_phase_count,
    expected_swap_count,
    expected_total_count,
):
    circuit = QFT(num_qubits).build_circuit()
    names = [operation.name for operation in circuit.operations]

    assert names.count("H") == expected_h_count
    assert names.count("C1-Phase") == expected_phase_count
    assert names.count("SWAP") == expected_swap_count
    assert len(names) == expected_total_count
    assert set(names) <= {"H", "C1-Phase", "SWAP"}
    assert circuit.used_symbols == []
    assert circuit.validate() is None


def test_build_returns_fresh_independent_circuits():
    transform = QFT(2)
    first = transform.build_circuit()
    second = transform.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_qft_composes_positionally_onto_non_contiguous_qubits():
    transform = QFT(3).build_circuit()
    destination = Circuit([7, 2, 9])

    destination.compose(transform, [Qubit(7), Qubit(2), Qubit(9)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2, 9]),
        transform.to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [True, 1.5, "2"])
def test_qft_rejects_non_integer_qubit_counts(num_qubits):
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        QFT(num_qubits)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_qft_rejects_non_positive_qubit_counts(num_qubits):
    with pytest.raises(ValueError, match="greater than zero"):
        QFT(num_qubits)


@pytest.mark.parametrize("degree", [True, 1.5, "1"])
def test_qft_rejects_non_integer_approximation_degrees(degree):
    with pytest.raises(TypeError, match="approximation_degree must be an integer"):
        QFT(3, approximation_degree=degree)  # type: ignore[arg-type]


@pytest.mark.parametrize("degree", [-1, 3, 4])
def test_qft_rejects_out_of_range_approximation_degrees(degree):
    with pytest.raises(ValueError, match=r"0 <= approximation_degree < num_qubits"):
        QFT(3, approximation_degree=degree)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("do_swaps", 1, "do_swaps must be a boolean"),
        ("inverse", 0, "inverse must be a boolean"),
    ],
)
def test_qft_rejects_non_boolean_flags(keyword, value, message):
    with pytest.raises(TypeError, match=message):
        QFT(2, **{keyword: value})


def test_qft_configuration_is_frozen_and_slotted():
    transform = QFT(2)

    assert not hasattr(transform, "__dict__")
    with pytest.raises(FrozenInstanceError):
        transform.num_qubits = 3  # type: ignore[misc]


def test_qft_supports_standard_dataclass_replace():
    transform = QFT(3)

    updated = replace(transform, approximation_degree=1, do_swaps=False)

    assert updated == QFT(3, approximation_degree=1, do_swaps=False)
    assert transform == QFT(3)
