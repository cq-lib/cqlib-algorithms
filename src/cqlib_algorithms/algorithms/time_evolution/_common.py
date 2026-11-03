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

"""Shared validation and trajectory evaluation for variational time evolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ..._utils.numeric import normalize_finite_real
from ...protocols import Estimator, EstimatorRequest, EstimatorResult
from ...protocols.estimator import require_estimator_results


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _snapshot_hamiltonian(
    operator: Hamiltonian | PauliString,
    *,
    name: str,
) -> Hamiltonian:
    """Return a defensive finite Hermitian Hamiltonian snapshot."""
    if isinstance(operator, PauliString):
        snapshot = Hamiltonian.from_pauli(operator.copy())
    elif isinstance(operator, Hamiltonian):
        snapshot = operator.copy()
    else:
        raise TypeError(f"{name} must be a Hamiltonian or PauliString")
    if snapshot.num_qubits <= 0:
        raise ValueError(f"{name} must act on at least one qubit")
    snapshot.simplify()
    # EstimatorRequest is the shared finite-Hermitian contract. Unlike
    # evolution-generator helpers, it intentionally permits zero Hamiltonians.
    EstimatorRequest(Circuit(snapshot.num_qubits), snapshot, {})
    return snapshot


def _snapshot_auxiliary_operators(
    operators: Mapping[str, Hamiltonian | PauliString] | None,
    *,
    num_qubits: int,
) -> Mapping[str, Hamiltonian]:
    """Return immutable snapshots of named auxiliary observables."""
    if operators is None:
        return MappingProxyType({})
    if not isinstance(operators, Mapping):
        raise TypeError("auxiliary_operators must be a mapping or None")

    snapshots: dict[str, Hamiltonian] = {}
    for name, operator in operators.items():
        if not isinstance(name, str):
            raise TypeError("auxiliary operator names must be strings")
        if not name:
            raise ValueError("auxiliary operator names must not be empty")
        snapshot = _snapshot_hamiltonian(
            operator,
            name=f"auxiliary_operators[{name!r}]",
        )
        if snapshot.num_qubits != num_qubits:
            raise ValueError("auxiliary operators must match the evolution width")
        snapshots[name] = snapshot
    return MappingProxyType(snapshots)


def _snapshot_parameters(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return a defensive immutable parameter snapshot."""
    if not isinstance(parameters, Mapping):
        raise TypeError("parameter history entries must be mappings")
    snapshot: dict[str, float] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise TypeError("parameter names must be strings")
        snapshot[name] = normalize_finite_real(value, "parameter value")
    return MappingProxyType(snapshot)


def _evaluate_trajectory(
    estimator: Estimator,
    circuits: Sequence[Circuit],
    hamiltonian: Hamiltonian,
    auxiliary_operators: Mapping[str, Hamiltonian],
    *,
    shots: int | None,
    seeds: Sequence[int],
) -> tuple[tuple[EstimatorResult, ...], Mapping[str, tuple[EstimatorResult, ...]]]:
    """Evaluate time-major trajectory observables and restore histories."""
    names = tuple(auxiliary_operators)
    observables = (hamiltonian, *auxiliary_operators.values())
    expected_count = len(circuits) * len(observables)
    if len(seeds) != expected_count:
        raise ValueError("trajectory seeds must match the observable request count")

    requests: list[EstimatorRequest] = []
    seed_index = 0
    for circuit in circuits:
        for observable in observables:
            requests.append(
                EstimatorRequest(circuit, observable, {}, shots, seeds[seed_index])
            )
            seed_index += 1

    results = require_estimator_results(
        estimator.run(requests),
        requests,
        count_error="estimator must return one result per trajectory request",
    )

    energies: list[EstimatorResult] = []
    histories: dict[str, list[EstimatorResult]] = {name: [] for name in names}
    stride = len(observables)
    for time_index in range(len(circuits)):
        offset = time_index * stride
        energies.append(results[offset])
        for observable_index, name in enumerate(names, start=1):
            histories[name].append(results[offset + observable_index])
    immutable_histories = MappingProxyType(
        {name: tuple(history) for name, history in histories.items()}
    )
    return tuple(energies), immutable_histories
