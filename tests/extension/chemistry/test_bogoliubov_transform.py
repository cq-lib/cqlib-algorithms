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

"""Independent conjugation tests for structured Bogoliubov circuits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqlib_algorithms.extension.chemistry import BogoliubovTransform


def _ladders(num_modes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    annihilation: list[np.ndarray] = []
    for mode in range(num_modes):
        matrix = np.zeros((1 << num_modes, 1 << num_modes), dtype=np.complex128)
        for basis in range(1 << num_modes):
            if (basis >> mode) & 1:
                sign = -1 if (basis & ((1 << mode) - 1)).bit_count() % 2 else 1
                matrix[basis ^ (1 << mode), basis] = sign
        annihilation.append(matrix)
    return [matrix.conj().T for matrix in annihilation], annihilation


def _majorana_to_w(orthogonal: np.ndarray) -> np.ndarray:
    even_even = orthogonal[0::2, 0::2]
    even_odd = orthogonal[0::2, 1::2]
    odd_even = orthogonal[1::2, 0::2]
    odd_odd = orthogonal[1::2, 1::2]
    creation = 0.5 * (even_even + odd_odd + 1j * (even_odd - odd_even))
    annihilation = 0.5 * (even_even - odd_odd - 1j * (even_odd + odd_even))
    return np.concatenate((creation, annihilation), axis=1)


def _assert_active_conjugation(transformation: np.ndarray) -> None:
    component = BogoliubovTransform(transformation)
    canonical = component.transformation
    num_modes = component.num_modes
    creation, annihilation = _ladders(num_modes)
    unitary = component.build_circuit().to_matrix()
    np.testing.assert_allclose(
        unitary.conj().T @ unitary,
        np.eye(1 << num_modes),
        rtol=0.0,
        atol=1e-12,
    )
    for index in range(num_modes):
        expected = np.zeros_like(creation[index])
        for mode in range(num_modes):
            expected += canonical[index, mode] * creation[mode]
            expected += canonical[index, num_modes + mode] * annihilation[mode]
        actual = unitary @ creation[index] @ unitary.conj().T
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def _plane(width: int, left: int, right: int, angle: float) -> np.ndarray:
    result = np.eye(width)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result[np.ix_([left, right], [left, right])] = [
        [cosine, sine],
        [-sine, cosine],
    ]
    return result


def test_identity_and_number_conserving_shorthand():
    _assert_active_conjugation(np.eye(3, dtype=np.complex128))
    rng = np.random.default_rng(247)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary, _ = np.linalg.qr(raw)
    _assert_active_conjugation(unitary)


def test_pure_particle_hole_transform_and_parity_branch():
    num_modes = 3
    transformation = np.concatenate(
        (np.zeros((num_modes, num_modes)), np.eye(num_modes)),
        axis=1,
    )
    component = BogoliubovTransform(transformation)
    assert component.changes_parity is True
    _assert_active_conjugation(transformation)


def test_true_pairing_and_noncommuting_majorana_planes_lock_order_and_signs():
    first = _plane(6, 0, 3, 0.41)
    second = _plane(6, 3, 5, -0.28)
    third = _plane(6, 1, 3, 0.33)
    orthogonal = first @ second @ third
    transformation = _majorana_to_w(orthogonal)
    assert np.max(np.abs(transformation[:, 3:])) > 0.1
    _assert_active_conjugation(transformation)
    reverse = third @ second @ first
    assert np.max(np.abs(orthogonal - reverse)) > 0.05


@pytest.mark.parametrize("num_modes", [1, 2, 3, 4])
def test_random_canonical_transformations_on_complete_fock_space(num_modes):
    rng = np.random.default_rng(771 + num_modes)
    for sample in range(5):
        raw = rng.normal(size=(2 * num_modes, 2 * num_modes))
        orthogonal, _ = np.linalg.qr(raw)
        if sample % 2:
            orthogonal[:, 0] *= -1.0
        transformation = _majorana_to_w(orthogonal)
        component = BogoliubovTransform(transformation)
        assert component.changes_parity is bool(np.linalg.det(orthogonal) < 0.0)
        _assert_active_conjugation(transformation)


def test_input_and_output_snapshots_and_repeated_builds_are_independent():
    transformation = np.eye(3, dtype=np.complex128)
    component = BogoliubovTransform(transformation)
    transformation[:] = 0.0
    exposed = component.transformation
    assert exposed.flags.writeable is False
    assert exposed is not component.transformation
    first = component.build_circuit()
    second = component.build_circuit()
    assert first is not second
    first.x(0)
    np.testing.assert_allclose(second.to_matrix(), np.eye(8), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("transformation", "error", "message"),
    [
        (1.0, ValueError, "two-dimensional"),
        ([1.0, 0.0], ValueError, "two-dimensional"),
        (np.zeros((0, 0)), ValueError, "at least one"),
        (np.zeros((2, 3)), ValueError, "shape"),
        ([[True]], TypeError, "booleans"),
        ([[float("nan")]], ValueError, "finite"),
        ([[1.0], [1.0, 2.0]], TypeError, "rectangular"),
        (np.diag([1.0, 2.0]), ValueError, "C C"),
        (
            np.array([[1.0, 0.0, 0.2, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            ValueError,
            "C C",
        ),
        (
            np.array([[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0]]),
            ValueError,
            "C C",
        ),
    ],
)
def test_invalid_transformations_are_rejected(transformation, error, message):
    with pytest.raises(error, match=message):
        BogoliubovTransform(transformation)
