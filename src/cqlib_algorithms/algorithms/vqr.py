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

"""Single-output quantum circuit learning with a mean-square objective.

References
----------
[1] K. Mitarai et al., "Quantum Circuit Learning", Phys. Rev. A 98,
    032309 (2018), arXiv:1803.00745v3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from cqlib import Circuit
from cqlib.qis import Hamiltonian, PauliString
from numpy.typing import ArrayLike, NDArray

from ..protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    _validate_hermitian,
    _validate_positive_shots,
)
from ..protocols.scalar_minimizer import ScalarMinimizer, ScalarMinimizerResult
from . import _ml_common as common
from . import _ml_validation as validation
from ._ml_validation import (
    _positive_integer,
    _real_hyperparameter,
    _targets,
    _validate_static_unitary_circuit,
)


@dataclass(frozen=True, slots=True)
class _State:
    parameters: Mapping[str, float]
    initial_loss: float
    training_loss: float
    optimization: ScalarMinimizerResult


class VQR:
    """Learn circuit weights by minimizing mean squared estimator predictions.

    Each input row produces one Hermitian observable expectation [1]. Qubit
    positions follow the circuit's insertion order. Fit retains the minimizer's
    selected point and independently evaluates its training MSE.

    Args:
        circuit: Static unitary data-and-weight circuit.
        input_parameters: Ordered names corresponding to input columns.
        weight_parameters: Ordered nonempty names optimized during training.
        observable: Hermitian observable with the circuit's qubit count.
        estimator: Batch expectation estimator.
        parameter_bindings: Fixed values for remaining circuit symbols.
        shots: Positive shots per sample, or None for exact expectations.
        optimizer: Scalar minimizer; None selects SciPy COBYLA.
        max_evaluations: Positive optimization-call budget. COBYLA requires
            at least the weight count plus two evaluations.
        tolerance: Positive final search scale for default COBYLA.
        initial_step: Positive initial search scale for default COBYLA.

    Raises:
        TypeError: If configuration types or component interfaces are invalid.
        ValueError: If symbols, dimensions, or numeric options are invalid.
    """

    def __init__(
        self,
        circuit: Circuit,
        input_parameters: Sequence[str],
        weight_parameters: Sequence[str],
        observable: Hamiltonian | PauliString,
        estimator: Estimator,
        *,
        parameter_bindings: Mapping[str, float] | None = None,
        shots: int | None = None,
        optimizer: ScalarMinimizer | None = None,
        max_evaluations: int = 200,
        tolerance: float = 1e-6,
        initial_step: float = 1.0,
    ) -> None:
        self._inputs = validation.names(input_parameters, "input_parameters")
        self._weights = validation.names(weight_parameters, "weight_parameters")
        self._fixed = validation.bindings(parameter_bindings, "parameter_bindings")
        self._circuit = validation.template(
            circuit, self._inputs, self._weights, self._fixed
        )
        if not isinstance(observable, (Hamiltonian, PauliString)):
            raise TypeError("observable must be a Hamiltonian or PauliString")
        _validate_hermitian(observable)
        if observable.num_qubits != self._circuit.num_qubits:
            raise ValueError("observable and circuit must have the same qubit count")
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        _validate_positive_shots(shots)
        self._observable = observable.copy()
        self._estimator = estimator
        self._shots = shots
        if optimizer is not None and not callable(optimizer):
            raise TypeError("optimizer must be callable")
        self._optimizer = optimizer
        if optimizer is None:
            self._tolerance, self._initial_step = common.configuration(
                max_evaluations, len(self._weights), tolerance, initial_step
            )
        else:
            _positive_integer(max_evaluations, "max_evaluations")
            self._tolerance = _real_hyperparameter(tolerance, "tolerance")
            self._initial_step = _real_hyperparameter(initial_step, "initial_step")
        self._max_evaluations = max_evaluations
        self._state: _State | None = None

    def _fitted(self) -> _State:
        if self._state is None:
            raise ValueError(
                "VQR must be fitted before prediction or fitted diagnostics"
            )
        return self._state

    @property
    def is_fitted(self) -> bool:
        """Return whether a fit has completed with a finite terminal point."""
        return self._state is not None

    @property
    def n_features_in_(self) -> int:
        """Return the input feature count after fit."""
        self._fitted()
        return len(self._inputs)

    @property
    def optimal_parameters_(self) -> Mapping[str, float]:
        """Return the immutable final weight bindings in declared order."""
        return self._fitted().parameters

    @property
    def initial_loss_(self) -> float:
        """Return the separately evaluated initial mean-square loss."""
        return self._fitted().initial_loss

    @property
    def training_loss_(self) -> float:
        """Return the separately evaluated terminal mean-square loss."""
        return self._fitted().training_loss

    @property
    def optimizer_success_(self) -> bool:
        """Return the minimizer's termination success flag."""
        return self._fitted().optimization.success

    @property
    def optimizer_status_(self) -> int:
        """Return the unaltered optimizer status code."""
        return self._fitted().optimization.status

    @property
    def optimizer_message_(self) -> str:
        """Return the optimizer termination message."""
        return self._fitted().optimization.message

    @property
    def optimizer_evaluations_(self) -> int:
        """Return actual optimization objective calls, bounded by configuration."""
        return self._fitted().optimization.evaluations

    @property
    def objective_evaluations_(self) -> int:
        """Return optimizer calls plus the initial and final diagnostic calls."""
        return self.optimizer_evaluations_ + 2

    def _predict(
        self, data: NDArray[np.float64], parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        requests: list[EstimatorRequest] = []
        # Validate the entire batch before any execution can occur.
        for row in data:
            bound = self._circuit.assign_parameters(
                {
                    **self._fixed,
                    **parameters,
                    **dict(zip(self._inputs, (float(value) for value in row))),
                }
            )
            _validate_static_unitary_circuit(bound, "bound VQR circuit")
            bound.inverse()
            requests.append(
                EstimatorRequest(bound, self._observable.copy(), {}, self._shots)
            )
        if not requests:
            return np.empty(0, dtype=np.float64)
        results = self._estimator.run(requests)
        if not isinstance(results, list):
            raise TypeError("estimator must return a list of EstimatorResult values")
        if len(results) != len(requests):
            raise ValueError("estimator must return one result per request")
        if any(not isinstance(result, EstimatorResult) for result in results):
            raise TypeError("estimator must return EstimatorResult values")
        prediction = np.array(
            [
                EstimatorResult(result.expectation, result.variance).expectation
                for result in results
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(prediction)):
            raise ValueError("estimator predictions must be finite")
        return prediction

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, initial_point: Mapping[str, float]
    ) -> VQR:
        """Fit nonempty samples and publish the selected weights and losses.

        Args:
            X: Finite real feature matrix with the declared input width.
            y: Matching one-dimensional real target vector.
            initial_point: Finite bindings covering all weight names.

        Returns:
            This model with updated fitted state after final loss evaluation.

        Raises:
            TypeError: If inputs or component results have invalid types.
            ValueError: If dimensions, values, or evaluation counts are invalid.
            ImportError: If default optimization requires an unavailable SciPy.
        """
        data = validation.data(X, len(self._inputs), nonempty=True)
        targets = _targets(y, len(data), "y")
        point = validation.initial(initial_point, self._weights)

        def objective(candidate: NDArray[np.float64]) -> float:
            return common.mse(
                self._predict(
                    data, validation.immutable_parameters(self._weights, candidate)
                ),
                targets,
            )

        initial_loss = objective(point)
        result = common.minimize(
            objective,
            point,
            self._max_evaluations,
            self._tolerance,
            self._initial_step,
            self._optimizer,
        )
        final_loss = objective(np.array(result.point, dtype=np.float64))
        self._state = _State(
            validation.immutable_parameters(self._weights, result.point),
            initial_loss,
            final_loss,
            result,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Evaluate observable expectations using the fitted weights.

        Args:
            X: Finite real feature matrix; shape (0,d) returns an empty vector.

        Returns:
            One prediction for each input row in its original order.

        Raises:
            TypeError: If data or estimator results have invalid types.
            ValueError: If the model is unfitted or values or shapes are invalid.
        """
        state = self._fitted()
        return self._predict(validation.data(X, len(self._inputs)), state.parameters)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute R-squared from at least two samples.

        Args:
            X: Finite real feature matrix with the declared input width.
            y: Matching finite one-dimensional targets.

        Returns:
            R-squared. Constant targets score one for an exact fit and zero
            otherwise.

        Raises:
            TypeError: If data or estimator results have invalid types.
            ValueError: If the model is unfitted, fewer than two targets are
                supplied, or values or shapes are invalid.
        """
        self._fitted()
        data = validation.data(X, len(self._inputs))
        targets = _targets(y, len(data), "y")
        if len(targets) < 2:
            raise ValueError("score requires at least two targets")
        numerator = common.mse(self.predict(data), targets)
        # Scaling the mean prevents a finite constant target from overflowing.
        scale = float(np.max(np.abs(targets)))
        mean = 0.0 if scale == 0.0 else float(np.mean(targets / scale)) * scale
        denominator = common.mse(np.full_like(targets, mean), targets)
        value = (
            1.0
            if numerator == 0.0
            else (0.0 if denominator == 0.0 else 1.0 - numerator / denominator)
        )
        if not math.isfinite(value):
            raise ValueError("R-squared must be finite")
        return value


__all__ = ["VQR"]
