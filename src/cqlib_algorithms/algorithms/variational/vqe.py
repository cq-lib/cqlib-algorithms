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

"""Variational quantum eigensolver composed from public Cqlib contracts.

References
----------
[1] A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum
    processor," Nature Communications, 5, 4213, 2014.
    DOI: 10.1038/ncomms5213.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

from cqlib.qis import Hamiltonian, PauliString

from ...protocols.ansatz import Ansatz
from ...protocols.estimator import Estimator, EstimatorRequest
from ...protocols.optimizer import Optimizer, OptimizerRequest, OptimizerResult
from .result import VQAResult

VQEResult = VQAResult


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

    Args:
        estimator: Component used to evaluate the objective.
        ansatz: Parameterized state-preparation component.
        optimizer: Classical optimizer for active parameters.
        prefix: Parameter-name prefix passed to the ansatz.

    Raises:
        TypeError: If a dependency or input has the wrong type.
        ValueError: If the prefix, operator, circuit, or initial point is invalid.
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

    def run(
        self,
        operator: Hamiltonian | PauliString,
        initial_point: Mapping[str, float],
        *,
        shots: int | None = None,
    ) -> VQEResult:
        """Run one VQE minimization and return the final optimizer iterate.

        Args:
            operator: Hermitian observable on the ansatz register.
            initial_point: Initial value of each active ansatz parameter,
                keyed by its name.
            shots: Optional sample count forwarded to the estimator.

        Returns:
            The optimizer's final energy, parameter bindings, bound circuit,
            and optimization result.

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

    def compute_minimum_eigenvalue(
        self,
        operator: Hamiltonian | PauliString,
        initial_point: Mapping[str, float],
        *,
        shots: int | None = None,
    ) -> VQEResult:
        """Run VQE with the same inputs and result as :meth:`run`."""
        return self.run(operator, initial_point, shots=shots)
