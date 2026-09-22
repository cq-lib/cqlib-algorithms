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

"""Variational quantum eigensolver composed from public Cqlib contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import final

from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString

from ..protocols.ansatz import Ansatz
from ..protocols.estimator import Estimator, EstimatorRequest
from ..protocols.optimizer import Optimizer, OptimizerRequest, OptimizerResult


def _immutable_parameters(parameters: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(parameters, Mapping):
        raise TypeError("optimal_parameters must be a mapping")

    normalized: dict[str, float] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise TypeError("optimal parameter names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("optimal parameter values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("optimal parameter values must be finite")
        normalized[name] = numeric_value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class VQEResult:
    """A VQE energy, optimal point, bound circuit and optimizer metadata.

    ``optimal_parameters`` preserves the optimizer's parameter order and is
    defensively copied into an immutable mapping. ``optimal_circuit`` is a
    separate, fully bound circuit; like other Core objects in this package it
    remains mutable even though this result dataclass is frozen.
    """

    eigenvalue: float
    optimal_parameters: Mapping[str, float]
    optimal_circuit: Circuit
    optimizer_result: OptimizerResult = field(repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.eigenvalue, bool) or not isinstance(self.eigenvalue, Real):
            raise TypeError("eigenvalue must be a real number")
        eigenvalue = float(self.eigenvalue)
        if not math.isfinite(eigenvalue):
            raise ValueError("eigenvalue must be finite")
        object.__setattr__(self, "eigenvalue", eigenvalue)

        if not isinstance(self.optimizer_result, OptimizerResult):
            raise TypeError("optimizer_result must be an OptimizerResult")
        parameters = _immutable_parameters(self.optimal_parameters)
        object.__setattr__(self, "optimal_parameters", parameters)

        names = tuple(parameters)
        if names != self.optimizer_result.parameter_names:
            raise ValueError("optimal parameter names must match the optimizer result")
        if tuple(parameters.values()) != self.optimizer_result.optimal_parameters:
            raise ValueError("optimal parameter values must match the optimizer result")
        if eigenvalue != self.optimizer_result.optimal_value:
            raise ValueError("eigenvalue must match the optimizer result")

        if not isinstance(self.optimal_circuit, Circuit):
            raise TypeError("optimal_circuit must be a cqlib.Circuit")
        self.optimal_circuit.validate()
        if self.optimal_circuit.used_symbols:
            raise ValueError("optimal_circuit must be fully bound")


@final
class VQE:
    """Minimize a Hermitian observable over a parameterized ansatz.

    For an ansatz state ``|psi(theta)>``, VQE minimizes
    ``<psi(theta)|H|psi(theta)>``. The estimator defines exact or finite-shot
    objective evaluation, while the injected optimizer owns the update rule
    and any gradient dependency.

    The initial point is explicit and must bind exactly the active symbols of
    the circuit returned by ``ansatz.build_circuit(prefix)``. Active symbol
    order comes from ``Circuit.used_symbols``; logical ansatz parameter slots
    that are absent from a degenerate circuit are not invented.
    """

    __slots__ = ("_ansatz", "_estimator", "_optimizer", "_prefix")

    def __init__(
        self,
        estimator: Estimator,
        ansatz: Ansatz,
        optimizer: Optimizer,
        *,
        prefix: str = "theta",
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")

        self._estimator = estimator
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._prefix = prefix

    @property
    def estimator(self) -> Estimator:
        """The injected expectation estimator."""
        return self._estimator

    @property
    def ansatz(self) -> Ansatz:
        """The variational circuit component."""
        return self._ansatz

    @property
    def optimizer(self) -> Optimizer:
        """The injected classical optimizer."""
        return self._optimizer

    @property
    def prefix(self) -> str:
        """The prefix used to build ansatz symbols."""
        return self._prefix

    def compute_minimum_eigenvalue(
        self,
        operator: Hamiltonian | PauliString,
        initial_point: Mapping[str, float],
        *,
        shots: int | None = None,
    ) -> VQEResult:
        """Run one VQE minimization and return the final optimizer iterate.

        The method delegates circuit construction, objective validation,
        expectation evaluation, differentiation and parameter updates to the
        configured public components. It performs no additional terminal
        estimator call: ``eigenvalue`` is the optimizer's independently
        evaluated ``optimal_value``.

        Raises:
            TypeError: If an input, dependency, or returned result has the
                wrong type.
            ValueError: If the observable, bindings, shots, or optimizer
                response violates a component contract.
        """
        circuit = self._ansatz.build_circuit(self._prefix)
        parameter_names = tuple(circuit.used_symbols)
        estimator_request = EstimatorRequest(
            circuit,
            operator,
            initial_point,
            shots,
        )
        optimizer_request = OptimizerRequest(estimator_request, parameter_names)

        optimizer_results = self._optimizer.run(
            self._estimator,
            [optimizer_request],
        )
        if not isinstance(optimizer_results, list):
            raise TypeError("optimizer must return a list of OptimizerResult values")
        if len(optimizer_results) != 1:
            raise ValueError(
                "optimizer must return exactly one result for the VQE request, "
                f"got {len(optimizer_results)}"
            )
        optimizer_result = optimizer_results[0]
        if not isinstance(optimizer_result, OptimizerResult):
            raise TypeError("optimizer must return OptimizerResult values")
        if optimizer_result.parameter_names != parameter_names:
            raise ValueError(
                "optimizer result parameter names must match the VQE request"
            )

        parameters = dict(
            zip(
                optimizer_result.parameter_names,
                optimizer_result.optimal_parameters,
                strict=True,
            )
        )
        optimal_circuit = circuit.assign_parameters(parameters)
        optimal_circuit.validate()
        if optimal_circuit.used_symbols:
            raise ValueError("optimizer result did not fully bind the ansatz circuit")

        return VQEResult(
            optimizer_result.optimal_value,
            parameters,
            optimal_circuit,
            optimizer_result,
        )
