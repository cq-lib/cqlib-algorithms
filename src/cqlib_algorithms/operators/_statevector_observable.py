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

"""Matrix-free Pauli action and expectation on statevector amplitudes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from cqlib.qis import Hamiltonian, PauliString


def apply_pauli(
    amplitudes: Sequence[complex] | np.ndarray,
    pauli: PauliString,
) -> np.ndarray:
    """Return the matrix-free Pauli action on a statevector."""
    state = np.asarray(amplitudes, dtype=np.complex128)
    indices = np.arange(state.size)
    # The explicit parity array avoids relying on NumPy-version-specific
    # vectorized population-count support.
    signs = np.asarray(
        [
            -1.0 if ((int(index) & pauli.z_mask).bit_count() & 1) else 1.0
            for index in indices
        ],
        dtype=np.float64,
    )
    result = np.empty_like(state)
    result[indices ^ pauli.x_mask] = state * signs
    result *= pauli.phase.to_complex() * pauli.y_phase()
    return result


def apply_observable(
    amplitudes: Sequence[complex] | np.ndarray,
    observable: Hamiltonian | PauliString,
) -> np.ndarray:
    """Return the matrix-free observable action on a statevector."""
    state = np.asarray(amplitudes, dtype=np.complex128)
    if isinstance(observable, PauliString):
        return apply_pauli(state, observable)
    result = np.zeros_like(state)
    for pauli, coefficient in observable.terms:
        result += coefficient * apply_pauli(state, pauli)
    return result


def pauli_expectation(
    amplitudes: Sequence[complex] | np.ndarray,
    pauli: PauliString,
) -> complex:
    """Return a matrix-free complex Pauli expectation."""
    state = np.asarray(amplitudes, dtype=np.complex128)
    return complex(np.vdot(state, apply_pauli(state, pauli)))


def hermitian_expectation(
    amplitudes: Sequence[complex] | np.ndarray,
    observable: Hamiltonian | PauliString,
) -> float:
    """Return a finite real expectation of a validated Hermitian observable."""
    state = np.asarray(amplitudes, dtype=np.complex128)
    observable_state = apply_observable(state, observable)
    return hermitian_expectation_from_action(state, observable, observable_state)


def hermitian_expectation_from_action(
    amplitudes: Sequence[complex] | np.ndarray,
    observable: Hamiltonian | PauliString,
    observable_state: Sequence[complex] | np.ndarray,
) -> float:
    """Return a Hermitian expectation from a precomputed observable action."""
    state = np.asarray(amplitudes, dtype=np.complex128)
    action = np.asarray(observable_state, dtype=np.complex128)
    if state.shape != action.shape:
        raise ValueError("observable action must match the statevector shape")
    value = complex(np.vdot(state, action))
    if isinstance(observable, PauliString):
        coefficient_scale = 1.0
    else:
        coefficient_scale = 1.0 + sum(
            abs(coefficient) for _, coefficient in observable.terms
        )
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("observable expectation must be finite")
    if abs(value.imag) > 1e-12 * coefficient_scale:
        raise ValueError("observable expectation has a non-negligible imaginary part")
    return float(value.real)
