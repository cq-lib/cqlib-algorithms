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

"""Deterministic statevector expectation estimation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector

from cqlib_algorithms.protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)

from ._circuit_validation import validate_unitary_state_preparation


def _pauli_expectation(amplitudes: Sequence[complex], pauli: PauliString) -> complex:
    """Compute a Pauli expectation in O(2**n) without a dense matrix.

    Core's current ``Statevector.expectation`` uses the wrong sign for an odd
    number of Y factors. The Core PauliString symplectic masks, phase and
    Y phase instead define the correct action on basis index ``j``:

    ``P|j> = phase * y_phase * (-1)**popcount(j & z_mask) |j ^ x_mask>``.

    The amplitude array comes from Core's ``Statevector.data`` binding,
    which copies the state once per request. No dense operator is allocated.
    """
    x_mask = pauli.x_mask
    z_mask = pauli.z_mask
    overlap = 0.0j
    for index, amplitude in enumerate(amplitudes):
        sign = -1 if ((index & z_mask).bit_count() & 1) else 1
        overlap += amplitudes[index ^ x_mask].conjugate() * amplitude * sign
    return complex(pauli.phase.to_complex() * pauli.y_phase() * overlap)


def _expectation(state: Statevector, observable: Hamiltonian | PauliString) -> float:
    """Return a real Hermitian expectation from the exact Core state."""
    amplitudes = cast(Sequence[complex], state.data)
    if isinstance(observable, PauliString):
        value = _pauli_expectation(amplitudes, observable)
        coefficient_scale = 1.0
    else:
        value = 0.0j
        coefficient_scale = 1.0
        for pauli, coefficient in observable.terms:
            value += coefficient * _pauli_expectation(amplitudes, pauli)
            coefficient_scale += abs(coefficient)

    # Request validation proves algebraic Hermiticity. Only floating-point
    # cancellation may leave a small imaginary component in the numeric sum.
    if not math.isfinite(value.imag) or abs(value.imag) > 1e-12 * coefficient_scale:
        raise ValueError("observable expectation has a non-negligible imaginary part")
    return float(value.real)


class StatevectorEstimator(Estimator):
    """Evaluate finite Hermitian observables on exact pure states.

    For each request this estimator computes
    ``<0|U(bound)^dagger observable U(bound)|0>`` using the Core
    :class:`cqlib.qis.state.Statevector` engine and Core Pauli symplectic
    data. The current Core expectation routine has a verified Y-sign defect,
    so this component evaluates each Pauli with the documented, matrix-free
    O(2**n) compatibility path instead. Observable position ``i``
    corresponds to ``request.circuit.qubits[i]`` in insertion order, even
    when Core qubit identifiers are non-contiguous or reordered. Results
    have deterministic estimator variance ``0.0``; this is not the physical
    variance of a single observable measurement.

    ``shots=None`` is required. The circuit must contain only unitary gates
    and optional barriers; measurement, reset, delay, and dynamic classical
    operations are rejected, including those nested inside circuit gates.
    Requests are revalidated immediately before each evaluation. Empty
    batches return an empty list, and result order matches request order.
    The Core objects in requests remain mutable but are never modified here.

    Raises:
        TypeError: If a batch entry is not an EstimatorRequest.
        ValueError: If a request asks for shots or uses non-unitary execution.
        cqlib.circuit.CircuitError: If Core circuit validation or simulation
            fails for another reason.
    """

    __slots__ = ()

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        """Return exact expectation values for all requests in input order."""
        results: list[EstimatorResult] = []
        for request in requests:
            if not isinstance(request, EstimatorRequest):
                raise TypeError("requests must contain EstimatorRequest instances")
            request.validate()
            if request.shots is not None:
                raise ValueError("shot-based estimation is not supported")

            validate_unitary_state_preparation(request.circuit)
            bound = request.circuit.assign_parameters(dict(request.parameter_bindings))
            state = Statevector.from_circuit(bound)
            results.append(
                EstimatorResult(_expectation(state, request.observable), 0.0)
            )
        return results
