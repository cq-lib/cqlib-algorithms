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

"""Tests for the Long phase-matched diffusion operator."""

from __future__ import annotations

import math
from dataclasses import is_dataclass

import numpy as np
import pytest
from cqlib import Circuit, Qubit

from cqlib_algorithms.operators.oracle import GroverDiffusion, GroverLong


def _long_reference(num_qubits: int, phase: float) -> np.ndarray:
    dimension = 1 << num_qubits
    uniform = np.ones((dimension, 1), dtype=np.complex128) / np.sqrt(dimension)
    projector = uniform @ uniform.conj().T
    return (
        np.eye(dimension, dtype=np.complex128) + (np.exp(1j * phase) - 1.0) * projector
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
@pytest.mark.parametrize("phase", [0.0, math.pi / 3.0, math.pi, -math.pi / 2.0])
def test_grover_long_matches_reference_matrix(num_qubits: int, phase: float) -> None:
    matrix = GroverLong(num_qubits, phase=phase).build_circuit().to_matrix()
    np.testing.assert_allclose(
        matrix, _long_reference(num_qubits, phase), rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_grover_long_with_pi_phase_is_negative_diffusion(num_qubits: int) -> None:
    actual = GroverLong(num_qubits, phase=math.pi).build_circuit().to_matrix()
    expected = -GroverDiffusion(num_qubits).build_circuit().to_matrix()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("phase", [0.0, 0.7, math.pi, -0.3])
def test_grover_long_is_unitary(num_qubits: int, phase: float) -> None:
    matrix = GroverLong(num_qubits, phase=phase).build_circuit().to_matrix()
    identity = np.eye(1 << num_qubits, dtype=np.complex128)
    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)


def test_build_returns_fresh_independent_circuits() -> None:
    operator = GroverLong(2, phase=0.5)
    first = operator.build_circuit()
    second = operator.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_grover_long_composes_onto_non_contiguous_qubits() -> None:
    operator = GroverLong(2, phase=0.5).build_circuit()
    destination = Circuit([7, 2])
    destination.compose(operator, [Qubit(7), Qubit(2)])

    np.testing.assert_allclose(
        destination.to_matrix([7, 2]),
        _long_reference(2, 0.5),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [True, 1.5, "2"])
def test_grover_long_rejects_non_integer_counts(num_qubits) -> None:
    with pytest.raises(TypeError, match="num_qubits must be an integer"):
        GroverLong(num_qubits, phase=math.pi)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_qubits", [0, -1])
def test_grover_long_rejects_non_positive_counts(num_qubits: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        GroverLong(num_qubits, phase=math.pi)


@pytest.mark.parametrize("phase", [True, "pi", complex(1.0, 0.0), math.inf, math.nan])
def test_grover_long_rejects_non_real_finite_phase(phase) -> None:
    with pytest.raises(TypeError, match="phase must be a real finite scalar"):
        GroverLong(2, phase=phase)  # type: ignore[arg-type]


def test_grover_long_is_a_slotted_component_with_read_only_config() -> None:
    operator = GroverLong(2, phase=0.5)

    assert not is_dataclass(operator)
    assert not hasattr(operator, "__dict__")
    assert operator.num_qubits == 2
    assert operator.phase == pytest.approx(0.5)
    with pytest.raises(AttributeError):
        operator.num_qubits = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        operator.phase = 1.0  # type: ignore[misc]


def test_helper_formula_chain_matches_total_phase() -> None:
    beta = GroverLong.get_beta(1, 8)
    it_num = GroverLong.get_it_num(beta)
    phi = GroverLong.get_phi(it_num, beta)
    total = GroverLong.get_phi_total(1, 8)

    assert beta == pytest.approx(math.asin(math.sqrt(1.0 / 8.0)))
    assert it_num == 1
    assert phi == pytest.approx(total)


@pytest.mark.parametrize("num_find", [True, 1.5, "1"])
def test_get_beta_rejects_invalid_num_find(num_find) -> None:
    with pytest.raises(TypeError, match="num_find must be an integer"):
        GroverLong.get_beta(num_find, 8)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_total", [True, 8.0, "8"])
def test_get_beta_rejects_invalid_num_total(num_total) -> None:
    with pytest.raises(TypeError, match="num_total must be an integer"):
        GroverLong.get_beta(1, num_total)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("num_find", "num_total", "message"),
    [
        (0, 8, "num_find must be greater than zero"),
        (-1, 8, "num_find must be greater than zero"),
        (1, 0, "num_total must be greater than zero"),
        (1, -1, "num_total must be greater than zero"),
        (5, 4, "num_find must not exceed num_total"),
    ],
)
def test_get_beta_rejects_invalid_value_ranges(
    num_find: int,
    num_total: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GroverLong.get_beta(num_find, num_total)


@pytest.mark.parametrize("beta", [True, "0.5", complex(0.5, 0.0), math.inf, math.nan])
def test_get_it_num_rejects_non_real_finite_beta(beta) -> None:
    with pytest.raises(TypeError, match="beta must be a real finite scalar"):
        GroverLong.get_it_num(beta)  # type: ignore[arg-type]


@pytest.mark.parametrize("beta", [0.0, -0.1, math.pi / 2.0 + 0.01, math.pi])
def test_get_it_num_rejects_beta_out_of_range(beta: float) -> None:
    with pytest.raises(ValueError, match=r"0 < beta <= pi / 2"):
        GroverLong.get_it_num(beta)


@pytest.mark.parametrize("it_num", [True, 1.5, "2"])
def test_get_phi_rejects_invalid_it_num(it_num) -> None:
    with pytest.raises(TypeError, match="it_num must be an integer"):
        GroverLong.get_phi(it_num, math.pi / 6.0)  # type: ignore[arg-type]


def test_get_phi_rejects_negative_it_num() -> None:
    with pytest.raises(ValueError, match="it_num must be non-negative"):
        GroverLong.get_phi(-1, math.pi / 6.0)


@pytest.mark.parametrize("beta", [True, "0.5", complex(0.5, 0.0), math.inf, math.nan])
def test_get_phi_rejects_non_real_finite_beta(beta) -> None:
    with pytest.raises(TypeError, match="beta must be a real finite scalar"):
        GroverLong.get_phi(1, beta)  # type: ignore[arg-type]


@pytest.mark.parametrize("beta", [0.0, -0.1, math.pi / 2.0 + 0.01, math.pi])
def test_get_phi_rejects_beta_out_of_range(beta: float) -> None:
    with pytest.raises(ValueError, match=r"0 < beta <= pi / 2"):
        GroverLong.get_phi(1, beta)


def test_get_phi_clamps_ratio_above_one_to_pi() -> None:
    phi = GroverLong.get_phi(0, math.pi / 12.0)
    assert phi == pytest.approx(math.pi)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_grover_long_with_zero_phase_is_identity(num_qubits: int) -> None:
    matrix = GroverLong(num_qubits, phase=0.0).build_circuit().to_matrix()
    np.testing.assert_allclose(
        matrix,
        np.eye(1 << num_qubits, dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


def test_grover_long_composes_onto_non_contiguous_qubits_in_ascending_order() -> None:
    operator = GroverLong(2, phase=0.5).build_circuit()
    destination = Circuit([2, 7])
    destination.compose(operator, [Qubit(2), Qubit(7)])

    np.testing.assert_allclose(
        destination.to_matrix([2, 7]),
        _long_reference(2, 0.5),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("num_find", "num_total", "expected_it_num"),
    [
        (1, 2, 0),
        (1, 4, 1),
        (1, 8, 1),
        (2, 8, 1),
        (1, 16, 2),
        (4, 16, 1),
    ],
)
def test_helper_formula_chain_matches_total_phase_for_multiple_ratios(
    num_find: int,
    num_total: int,
    expected_it_num: int,
) -> None:
    beta = GroverLong.get_beta(num_find, num_total)
    it_num = GroverLong.get_it_num(beta)
    phi = GroverLong.get_phi(it_num, beta)
    total = GroverLong.get_phi_total(num_find, num_total)

    assert math.isfinite(beta)
    assert math.isfinite(phi)
    assert math.isfinite(total)
    assert it_num == expected_it_num
    assert phi == pytest.approx(total)
