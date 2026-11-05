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

# Modified to use the shared fidelity quantum-kernel operator.

"""Projected, unbiased k=1 kernel Pegasos binary classification.

Reference: Shalev-Shwartz, Singer and Srebro, ICML 2007, Figure 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._utils.numeric import copy_finite_real_array, validate_positive_integer
from ..operators.quantum_kernel import FidelityQuantumKernel
from . import _ml_common as common
from . import _ml_validation as validation
from ._ml_validation import _real_hyperparameter, _targets


@dataclass(frozen=True, slots=True)
class _State:
    data: NDArray[np.float64]
    classes: tuple[float, float]
    coefficients: NDArray[np.float64]
    raw: NDArray[np.float64]
    matrix: NDArray[np.float64]
    minimum: float
    correction: float
    norm: float
    objective: float


def _norm_squared(
    coefficients: NDArray[np.float64], matrix: NDArray[np.float64]
) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = float(coefficients @ matrix @ coefficients)
            scale = float(np.abs(coefficients) @ np.abs(matrix) @ np.abs(coefficients))
        except FloatingPointError as error:
            raise ValueError("kernel norm computation overflowed") from error
    if not math.isfinite(value) or not math.isfinite(scale):
        raise ValueError("kernel norm must be finite")
    tolerance = 16.0 * np.finfo(np.float64).eps * len(coefficients) * scale
    if value < -tolerance:
        raise ValueError("kernel squared norm is significantly negative")
    return max(0.0, value)


class PegasosQSVC:
    """Train a binary SVM with projected single-sample Pegasos updates.

    The objective is lambda*||w||^2/2 + mean hinge loss. Each step samples
    one observation with a local RNG. Training uses the PSD positive part of
    the Gram matrix; prediction evaluates rectangular fidelities. Numeric
    labels are sorted, and a zero decision maps to the positive class.
    The returned coefficients represent the final iterate with zero bias.
    """

    def __init__(
        self,
        quantum_kernel: FidelityQuantumKernel,
        *,
        regularization: float = 0.1,
        num_steps: int = 1000,
        seed: int | None = None,
    ) -> None:
        if not isinstance(quantum_kernel, FidelityQuantumKernel):
            raise TypeError("quantum_kernel must be a FidelityQuantumKernel")
        self._regularization = _real_hyperparameter(regularization, "regularization")
        validate_positive_integer(num_steps, "num_steps")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        if seed is not None and seed < 0:
            raise ValueError("seed must be nonnegative")
        self._kernel = quantum_kernel
        self._num_steps = num_steps
        self._seed = seed
        self._state: _State | None = None

    def _fitted(self) -> _State:
        if self._state is None:
            raise ValueError(
                "PegasosQSVC must be fitted before prediction or fitted diagnostics"
            )
        return self._state

    @property
    def is_fitted(self) -> bool:
        """Return whether training has completed."""
        return self._state is not None

    @property
    def classes_(self) -> tuple[float, float]:
        """Return negative and positive original labels in sorted order."""
        return self._fitted().classes

    @property
    def n_features_in_(self) -> int:
        """Return the fitted feature width."""
        return int(self._fitted().data.shape[1])

    @property
    def training_data_(self) -> NDArray[np.float64]:
        """Return a copy of the retained training data."""
        return np.array(self._fitted().data, dtype=np.float64, copy=True)

    @property
    def coefficients_(self) -> NDArray[np.float64]:
        """Return copied signed coefficients in w=sum_i c_i phi(x_i)."""
        return np.array(self._fitted().coefficients, dtype=np.float64, copy=True)

    @property
    def raw_kernel_matrix_(self) -> NDArray[np.float64]:
        """Return a copy of training fidelities before repair."""
        return np.array(self._fitted().raw, dtype=np.float64, copy=True)

    @property
    def kernel_matrix_(self) -> NDArray[np.float64]:
        """Return a copy of the PSD positive part used for training."""
        return np.array(self._fitted().matrix, dtype=np.float64, copy=True)

    @property
    def minimum_raw_eigenvalue_(self) -> float:
        """Return the smallest raw training Gram eigenvalue."""
        return self._fitted().minimum

    @property
    def psd_correction_norm_(self) -> float:
        """Return the Frobenius norm of the training matrix correction."""
        return self._fitted().correction

    @property
    def weight_norm_(self) -> float:
        """Return the final weight norm in the training PSD feature space."""
        return self._fitted().norm

    @property
    def training_objective_(self) -> float:
        """Return regularization plus mean hinge loss at the final iterate."""
        return self._fitted().objective

    @property
    def n_iter_(self) -> int:
        """Return the performed update count."""
        self._fitted()
        return self._num_steps

    def fit(self, X: ArrayLike, y: ArrayLike) -> PegasosQSVC:
        """Fit finite nonempty X and exactly two numeric classes atomically."""
        data = validation.data(X, self._kernel.num_features, nonempty=True)
        labels, classes = validation.binary_labels(y, len(data))
        raw = common.gram(self._kernel.evaluate(data), len(data))
        eigenvalues, eigenvectors = np.linalg.eigh(raw)
        matrix = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        matrix = (matrix + matrix.T) / 2.0
        coefficients = np.zeros(len(data), dtype=np.float64)
        rng = np.random.default_rng(self._seed)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for step in range(1, self._num_steps + 1):
                    index = int(rng.integers(len(data)))
                    margin = float(labels[index] * (matrix[index] @ coefficients))
                    if not math.isfinite(margin):
                        raise ValueError("training margin must be finite")
                    coefficients *= 1.0 - 1.0 / step
                    if margin < 1.0:
                        coefficients[index] += (
                            labels[index] / self._regularization / step
                        )
                    norm_squared = _norm_squared(coefficients, matrix)
                    norm = math.sqrt(norm_squared)
                    radius = 1.0 / math.sqrt(self._regularization)
                    if norm > radius:
                        coefficients *= radius / norm
                norm_squared = _norm_squared(coefficients, matrix)
                objective = self._regularization * norm_squared / 2.0 + float(
                    np.mean(np.maximum(0.0, 1.0 - labels * (matrix @ coefficients)))
                )
            except (FloatingPointError, OverflowError) as error:
                raise ValueError("Pegasos arithmetic overflowed") from error
        if not math.isfinite(objective):
            raise ValueError("training objective must be finite")
        self._state = _State(
            data,
            classes,
            coefficients,
            raw,
            matrix,
            float(eigenvalues[0]),
            float(np.linalg.norm(matrix - raw, ord="fro")),
            math.sqrt(norm_squared),
            objective,
        )
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return raw rectangular kernel decisions; empty (0,d) performs no work."""
        state = self._fitted()
        data = validation.data(X, self._kernel.num_features)
        if not len(data):
            return np.empty(0, dtype=np.float64)
        matrix = copy_finite_real_array(
            self._kernel.evaluate(data, state.data), "prediction kernel", 2
        )
        if (
            matrix.shape != (len(data), len(state.data))
            or np.any(matrix < 0.0)
            or np.any(matrix > 1.0)
        ):
            raise ValueError("kernel must return rectangular fidelities within [0, 1]")
        with np.errstate(over="raise", invalid="raise"):
            try:
                scores = matrix @ state.coefficients
            except FloatingPointError as error:
                raise ValueError("prediction arithmetic overflowed") from error
        if not np.all(np.isfinite(scores)):
            raise ValueError("prediction scores must be finite")
        return np.array(scores, dtype=np.float64, copy=True)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return original numeric labels; decision >= 0 chooses the positive class."""
        classes = self._fitted().classes
        return np.where(self.decision_function(X) >= 0.0, classes[1], classes[0])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return mean classification accuracy on nonempty known-class labels."""
        state = self._fitted()
        data = validation.data(X, self._kernel.num_features, nonempty=True)
        targets = _targets(y, len(data), "y")
        if not np.all(np.isin(targets, state.classes)):
            raise ValueError("score labels must belong to the fitted classes")
        return float(np.mean(self.predict(data) == targets))


__all__ = ["PegasosQSVC"]
