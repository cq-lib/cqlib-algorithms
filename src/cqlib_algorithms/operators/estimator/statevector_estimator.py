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

from collections.abc import Sequence

from cqlib.qis.state import Statevector

from cqlib_algorithms.protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
)

from .._statevector_observable import hermitian_expectation
from ._circuit_validation import validate_unitary_state_preparation


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
                EstimatorResult(
                    hermitian_expectation(state.data, request.observable),
                    0.0,
                )
            )
        return results
