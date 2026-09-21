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

"""Shared validation for expectation-estimator preparation circuits."""

from cqlib.circuit import CircuitError

from cqlib import Circuit


def validate_unitary_state_preparation(circuit: Circuit) -> None:
    """Reject dynamic and non-unitary operations, including nested gates."""
    if circuit.classical_vars or circuit.classical_values:
        raise ValueError("estimation requires a unitary circuit")

    try:
        expanded = circuit.decompose()
    except CircuitError as exc:
        raise ValueError("estimation requires a decomposable unitary circuit") from exc

    for operation in expanded.operations:
        if (
            operation.is_classical_data
            or operation.is_classical_control
            or operation.is_delay
            or (operation.is_directive and operation.name != "barrier")
        ):
            raise ValueError(
                "estimation requires a unitary circuit without measurement, "
                "reset, delay, or classical control"
            )
