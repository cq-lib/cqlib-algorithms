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
# Modified to compose trainable kernels, losses and scalar minimizers.

"""Composable quantum-kernel training and independent trained results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import numpy as np
from cqlib import Circuit
from numpy.typing import ArrayLike, NDArray

from ..._utils.numeric import copy_finite_real_array, validate_positive_integer
from ...operators.quantum_kernel import (
    BalancedKernelAlignmentLoss,
    FidelityQuantumKernel,
    TrainableFidelityQuantumKernel,
)
from ...protocols.quantum_kernel import (
    KernelLoss,
    QuantumKernel,
    TrainableQuantumKernel,
)
from ...protocols.sampler import Sampler
from ...protocols.scalar_minimizer import ScalarMinimizer, ScalarMinimizerResult
from ._common import configuration, gram, minimize
from ._validation import (
    _real_hyperparameter,
    _targets,
    bindings,
    immutable_parameters,
    initial,
    names,
)
from ._validation import (
    data as validate_data,
)


@dataclass(frozen=True, slots=True)
class QuantumKernelTrainerResult:
    """A trained kernel and the losses evaluated during one fit.

    Parameters are copied into an immutable mapping; the matrix is a read-only
    snapshot. The final loss and matrix share one evaluation. ``optimizer_result``
    retains the minimizer's observed loss at its selected point.
    """

    quantum_kernel: QuantumKernel
    optimal_parameters: Mapping[str, float]
    initial_loss: float
    loss: float
    kernel_matrix: NDArray[np.float64] = field(repr=False, compare=False)
    optimizer_result: ScalarMinimizerResult

    def __post_init__(self) -> None:
        if not isinstance(self.quantum_kernel, QuantumKernel):
            raise TypeError("quantum_kernel must be a QuantumKernel")
        parameters = bindings(self.optimal_parameters, "optimal_parameters")
        object.__setattr__(self, "optimal_parameters", MappingProxyType(parameters))
        for name in ("initial_loss", "loss"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        matrix = copy_finite_real_array(self.kernel_matrix, "kernel matrix", 2)
        if matrix.shape[0] != matrix.shape[1] or not np.array_equal(matrix, matrix.T):
            raise ValueError("kernel must return a symmetric square training matrix")
        snapshot: NDArray[np.float64] = np.frombuffer(
            matrix.tobytes(), dtype=np.float64
        ).reshape(matrix.shape)
        object.__setattr__(self, "kernel_matrix", snapshot)
        if not isinstance(self.optimizer_result, ScalarMinimizerResult):
            raise TypeError("optimizer_result must be a ScalarMinimizerResult")


class QuantumKernelTrainer:
    """Optimize a trainable kernel using a scalar loss and minimizer.

    Supply ``quantum_kernel`` or a circuit with input/trainable names.
    BalancedKernelAlignmentLoss and SciPy COBYLA are the defaults. A custom
    loss defines target semantics; a custom optimizer implements ScalarMinimizer.
    ``max_evaluations`` bounds optimization calls; initial and final diagnostic
    evaluations add two calls. Fit publishes an independent ``result_``.
    Invalid configuration and component outputs raise TypeError or ValueError.
    """

    def __init__(
        self,
        feature_map: Circuit | None = None,
        input_parameters: Sequence[str] | None = None,
        trainable_parameters: Sequence[str] | None = None,
        *,
        quantum_kernel: TrainableQuantumKernel | None = None,
        loss: KernelLoss | None = None,
        optimizer: ScalarMinimizer | None = None,
        parameter_bindings: Mapping[str, float] | None = None,
        sampler: Sampler | None = None,
        shots: int | None = None,
        max_statevector_qubits: int | None = None,
        max_statevector_bytes: int | None = None,
        max_evaluations: int = 200,
        tolerance: float = 1e-6,
        initial_step: float = 1.0,
    ) -> None:
        if quantum_kernel is None:
            if (
                feature_map is None
                or input_parameters is None
                or trainable_parameters is None
            ):
                raise ValueError(
                    "supply a feature_map with input and trainable parameters"
                )
            quantum_kernel = TrainableFidelityQuantumKernel(
                feature_map,
                input_parameters,
                trainable_parameters,
                parameter_bindings=parameter_bindings,
                sampler=sampler,
                shots=shots,
                max_statevector_qubits=20
                if max_statevector_qubits is None
                else max_statevector_qubits,
                max_statevector_bytes=268_435_456
                if max_statevector_bytes is None
                else max_statevector_bytes,
            )
        elif any(
            value is not None
            for value in (
                feature_map,
                input_parameters,
                trainable_parameters,
                parameter_bindings,
                sampler,
                shots,
                max_statevector_qubits,
                max_statevector_bytes,
            )
        ):
            raise ValueError(
                "quantum_kernel and circuit configuration are mutually exclusive"
            )
        if not isinstance(quantum_kernel, TrainableQuantumKernel):
            raise TypeError("quantum_kernel must be a TrainableQuantumKernel")
        validate_positive_integer(quantum_kernel.num_features, "num_features")
        self._trainables = names(
            quantum_kernel.trainable_parameters, "trainable_parameters"
        )
        self._width = quantum_kernel.num_features
        self._kernel_family = quantum_kernel
        if loss is not None and not callable(loss):
            raise TypeError("loss must be callable")
        if optimizer is not None and not callable(optimizer):
            raise TypeError("optimizer must be callable")
        self._loss: KernelLoss = BalancedKernelAlignmentLoss() if loss is None else loss
        self._optimizer = optimizer
        if optimizer is None:
            self._tolerance, self._initial_step = configuration(
                max_evaluations, len(self._trainables), tolerance, initial_step
            )
        else:
            validate_positive_integer(max_evaluations, "max_evaluations")
            self._tolerance = _real_hyperparameter(tolerance, "tolerance")
            self._initial_step = _real_hyperparameter(initial_step, "initial_step")
        self._max_evaluations = max_evaluations
        self._result: QuantumKernelTrainerResult | None = None
        self._classes: tuple[float, float] | None = None

    def _fitted(self) -> QuantumKernelTrainerResult:
        if self._result is None:
            raise ValueError(
                "QuantumKernelTrainer must be fitted before fitted diagnostics"
            )
        return self._result

    @property
    def result_(self) -> QuantumKernelTrainerResult:
        """Return the independently retained result of the latest fit."""
        return self._fitted()

    @property
    def is_fitted(self) -> bool:
        """Return whether a fit has completed."""
        return self._result is not None

    @property
    def quantum_kernel_(self) -> QuantumKernel:
        """Return the kernel at the optimizer-selected parameter point."""
        return self._fitted().quantum_kernel

    @property
    def optimal_parameters_(self) -> Mapping[str, float]:
        """Return immutable parameters in declared order."""
        return self._fitted().optimal_parameters

    @property
    def n_features_in_(self) -> int:
        """Return the fitted feature width."""
        self._fitted()
        return self._width

    @property
    def kernel_matrix_(self) -> NDArray[np.float64]:
        """Return an independent final Gram matrix."""
        return np.array(self._fitted().kernel_matrix, dtype=np.float64, copy=True)

    @property
    def initial_loss_(self) -> float:
        """Return the initial diagnostic loss."""
        return self._fitted().initial_loss

    @property
    def loss_(self) -> float:
        """Return the final diagnostic loss."""
        return self._fitted().loss

    @property
    def classes_(self) -> tuple[float, float]:
        """Return sorted binary labels for BalancedKernelAlignmentLoss."""
        self._fitted()
        if self._classes is None:
            raise ValueError("classes_ requires BalancedKernelAlignmentLoss")
        return self._classes

    @property
    def alignment_(self) -> float:
        """Return final alignment for BalancedKernelAlignmentLoss."""
        _ = self.classes_
        return -self.loss_

    @property
    def initial_alignment_(self) -> float:
        """Return initial alignment for BalancedKernelAlignmentLoss."""
        _ = self.classes_
        return -self.initial_loss_

    @property
    def optimizer_success_(self) -> bool:
        """Return the minimizer termination flag."""
        return self._fitted().optimizer_result.success

    @property
    def optimizer_status_(self) -> int:
        """Return the minimizer status code."""
        return self._fitted().optimizer_result.status

    @property
    def optimizer_message_(self) -> str:
        """Return the minimizer termination message."""
        return self._fitted().optimizer_result.message

    @property
    def optimizer_evaluations_(self) -> int:
        """Return actual optimizer objective calls."""
        return self._fitted().optimizer_result.evaluations

    @property
    def objective_evaluations_(self) -> int:
        """Return optimization calls plus two diagnostic evaluations."""
        return self.optimizer_evaluations_ + 2

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, initial_point: Mapping[str, float]
    ) -> QuantumKernelTrainer:
        """Fit finite feature rows and matching real targets, then return self.

        Initial bindings cover the declared trainable names exactly. The loss
        validates target semantics. Each candidate creates an independent
        bound kernel; successful final evaluation publishes a new result.
        """
        data = validate_data(X, self._width, nonempty=True)
        targets = _targets(y, len(data), "y")
        point = initial(initial_point, self._trainables)
        classes = (
            self._loss.classes(targets)
            if isinstance(self._loss, BalancedKernelAlignmentLoss)
            else None
        )

        def evaluate(
            candidate: NDArray[np.float64],
        ) -> tuple[QuantumKernel, NDArray[np.float64], float]:
            kernel = self._kernel_family.bind_parameters(
                immutable_parameters(self._trainables, candidate)
            )
            if not isinstance(kernel, QuantumKernel):
                raise TypeError("bind_parameters must return a QuantumKernel")
            if kernel.num_features != self._width:
                raise ValueError(
                    "bound kernel feature width must match trainable kernel"
                )
            values = kernel.evaluate(data.copy())
            if isinstance(kernel, FidelityQuantumKernel):
                matrix = gram(values, len(data))
            else:
                matrix = copy_finite_real_array(values, "kernel matrix", 2)
                if matrix.shape != (len(data), len(data)) or not np.array_equal(
                    matrix, matrix.T
                ):
                    raise ValueError(
                        "kernel must return a symmetric square training matrix"
                    )
            value = self._loss(matrix.copy(), targets.copy())
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("loss must return a real number")
            if not math.isfinite(float(value)):
                raise ValueError("loss must return a finite value")
            return kernel, matrix, float(value)

        def objective(candidate: NDArray[np.float64]) -> float:
            return evaluate(candidate)[2]

        initial_loss = objective(point)
        optimization = minimize(
            objective,
            point,
            self._max_evaluations,
            self._tolerance,
            self._initial_step,
            self._optimizer,
        )
        kernel, matrix, final_loss = evaluate(
            np.array(optimization.point, dtype=np.float64)
        )
        result = QuantumKernelTrainerResult(
            kernel,
            immutable_parameters(self._trainables, optimization.point),
            initial_loss,
            final_loss,
            matrix,
            optimization,
        )
        self._result = result
        self._classes = classes
        return self


__all__ = ["QuantumKernelTrainer", "QuantumKernelTrainerResult"]
