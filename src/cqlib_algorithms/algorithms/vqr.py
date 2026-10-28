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

"""Single-output quantum circuit learning with a true mean-square objective.

Reference: Mitarai et al., Phys. Rev. A 98, 032309 (2018), arXiv:1803.00745v3.
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
from . import _ml_common as common
from .amplitude_estimation.problem import _validate_static_unitary_circuit
from .qsvr import _targets


@dataclass(frozen=True, slots=True)
class _State:
    parameters: Mapping[str, float]
    initial_loss: float
    training_loss: float
    optimization: common.Optimization


class VQR:
    """Learn circuit weights by minimizing mean squared estimator predictions.

    Data, weight and fixed names must be disjoint and exactly cover the static
    circuit symbols. Each prediction is a Hermitian observable expectation;
    targets are never normalized or clipped. COBYLA is a lazy SciPy dependency.
    Its evaluation budget excludes one initial and one final diagnostic call,
    both included in ``objective_evaluations_``. Finite optimizer termination
    points remain usable even when ``optimizer_success_`` is false. Failed fits
    preserve the previous model. Exact execution uses ``shots=None``; a supplied
    positive shot count is forwarded to the injected Estimator.
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
        max_evaluations: int = 200,
        tolerance: float = 1e-6,
        initial_step: float = 1.0,
    ) -> None:
        self._inputs = common.names(input_parameters, "input_parameters")
        self._weights = common.names(weight_parameters, "weight_parameters")
        self._fixed = common.bindings(parameter_bindings, "parameter_bindings")
        self._circuit = common.template(
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
        self._tolerance, self._initial_step = common.configuration(
            max_evaluations, len(self._weights), tolerance, initial_step
        )
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
        """Return SciPy's termination success flag, not a global guarantee."""
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
        """Fit nonempty finite real X and one-dimensional matching targets.

        Initial bindings must cover exactly the weights. Singular expressions,
        invalid estimator/optimizer outputs and nonfinite losses raise errors;
        no partial fitted state is committed. SciPy absence raises ImportError.
        """
        data = common.data(X, len(self._inputs), nonempty=True)
        targets = _targets(y, len(data), "y")
        point = common.initial(initial_point, self._weights)

        def objective(candidate: NDArray[np.float64]) -> float:
            return common.mse(
                self._predict(
                    data, common.immutable_parameters(self._weights, candidate)
                ),
                targets,
            )

        initial_loss = objective(point)
        result = common.minimize(
            objective, point, self._max_evaluations, self._tolerance, self._initial_step
        )
        final_loss = objective(result.point)
        self._state = _State(
            common.immutable_parameters(self._weights, result.point),
            initial_loss,
            final_loss,
            result,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return final observable predictions; empty (0,d) causes no execution."""
        state = self._fitted()
        return self._predict(common.data(X, len(self._inputs)), state.parameters)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return R-squared for at least two targets; constant-target scores are 1/0."""
        self._fitted()
        data = common.data(X, len(self._inputs))
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
