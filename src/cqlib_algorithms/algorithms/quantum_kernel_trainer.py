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

"""Class-balanced, uncentered kernel-target alignment training.

Reference: Hubregtsen et al., arXiv:2105.02276v1, Section IV, Eq. (27).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from cqlib import Circuit
from numpy.typing import ArrayLike, NDArray

from ..operators.quantum_kernel import FidelityQuantumKernel
from ..protocols.sampler import Sampler
from . import _ml_common as common


@dataclass(frozen=True, slots=True)
class _State:
    kernel: FidelityQuantumKernel
    parameters: Mapping[str, float]
    classes: tuple[float, float]
    initial_alignment: float
    alignment: float
    matrix: NDArray[np.float64]
    optimization: common.Optimization


def _alignment(matrix: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    # Normalize the arrays before the inner product to avoid denominator overflow.
    kernel_norm = float(np.linalg.norm(matrix, ord="fro"))
    target_norm = float(np.linalg.norm(target, ord="fro"))
    if (
        not np.isfinite(kernel_norm)
        or not np.isfinite(target_norm)
        or kernel_norm <= 0.0
        or target_norm <= 0.0
    ):
        raise ValueError("alignment norms must be finite and positive")
    result = float(np.sum((matrix / kernel_norm) * (target / target_norm)))
    tolerance = 16.0 * np.finfo(np.float64).eps * matrix.size
    if not np.isfinite(result) or not -1.0 - tolerance <= result <= 1.0 + tolerance:
        raise ValueError("alignment must lie within [-1, 1]")
    return min(1.0, max(-1.0, result))


class QuantumKernelTrainer:
    """Train circuit parameters by minimizing negative balanced alignment.

    For binary +/- labels z, t_i=z_i/n_class(i) and T=t*t.T. The loss is
    -<K,T>_F/(||K||_F*||T||_F), without centering or PSD repair. Each candidate
    creates a fresh FidelityQuantumKernel through its public configuration.
    Training returns a usable fixed kernel, not predictions. Finite-shot losses
    are empirical and may fluctuate; improvement or global convergence is not
    guaranteed. Atomic fit and truthful COBYLA diagnostics match VQR.
    """

    def __init__(
        self,
        feature_map: Circuit,
        input_parameters: Sequence[str],
        trainable_parameters: Sequence[str],
        *,
        parameter_bindings: Mapping[str, float] | None = None,
        sampler: Sampler | None = None,
        shots: int | None = None,
        max_statevector_qubits: int = 20,
        max_statevector_bytes: int = 268_435_456,
        max_evaluations: int = 200,
        tolerance: float = 1e-6,
        initial_step: float = 1.0,
    ) -> None:
        self._inputs = common.names(input_parameters, "input_parameters")
        self._trainables = common.names(trainable_parameters, "trainable_parameters")
        self._fixed = common.bindings(parameter_bindings, "parameter_bindings")
        self._circuit = common.template(
            feature_map, self._inputs, self._trainables, self._fixed
        )
        # Let the existing kernel validate its own execution configuration while
        # leaving every trainable expression symbolic until an actual fit point.
        FidelityQuantumKernel(
            self._circuit,
            self._inputs + self._trainables,
            parameter_bindings=self._fixed,
            sampler=sampler,
            shots=shots,
            max_statevector_qubits=max_statevector_qubits,
            max_statevector_bytes=max_statevector_bytes,
        )
        self._sampler = sampler
        self._shots = shots
        self._max_statevector_qubits = max_statevector_qubits
        self._max_statevector_bytes = max_statevector_bytes
        self._tolerance, self._initial_step = common.configuration(
            max_evaluations, len(self._trainables), tolerance, initial_step
        )
        self._max_evaluations = max_evaluations
        self._state: _State | None = None

    def _fitted(self) -> _State:
        if self._state is None:
            raise ValueError(
                "QuantumKernelTrainer must be fitted before fitted diagnostics"
            )
        return self._state

    @property
    def is_fitted(self) -> bool:
        """Return whether training completed with a valid terminal kernel."""
        return self._state is not None

    @property
    def quantum_kernel_(self) -> FidelityQuantumKernel:
        """Return the frozen-configuration kernel for downstream learning."""
        return self._fitted().kernel

    @property
    def optimal_parameters_(self) -> Mapping[str, float]:
        """Return immutable final trainable bindings in declared order."""
        return self._fitted().parameters

    @property
    def classes_(self) -> tuple[float, float]:
        """Return the two original classes in sorted order."""
        return self._fitted().classes

    @property
    def n_features_in_(self) -> int:
        """Return fitted feature width."""
        self._fitted()
        return len(self._inputs)

    @property
    def kernel_matrix_(self) -> NDArray[np.float64]:
        """Return copied raw Gram from the terminal diagnostic evaluation."""
        return np.array(self._fitted().matrix, dtype=np.float64, copy=True)

    @property
    def initial_alignment_(self) -> float:
        """Return separately evaluated initial alignment."""
        return self._fitted().initial_alignment

    @property
    def alignment_(self) -> float:
        """Return alignment from the same evaluation as kernel_matrix_."""
        return self._fitted().alignment

    @property
    def loss_(self) -> float:
        """Return negative terminal alignment."""
        return -self._fitted().alignment

    @property
    def optimizer_success_(self) -> bool:
        """Return SciPy's local termination success flag."""
        return self._fitted().optimization.success

    @property
    def optimizer_status_(self) -> int:
        """Return the original optimizer status code."""
        return self._fitted().optimization.status

    @property
    def optimizer_message_(self) -> str:
        """Return the optimizer termination message."""
        return self._fitted().optimization.message

    @property
    def optimizer_evaluations_(self) -> int:
        """Return actual optimizer objective evaluations."""
        return self._fitted().optimization.evaluations

    @property
    def objective_evaluations_(self) -> int:
        """Return optimizer evaluations plus initial and final diagnostics."""
        return self.optimizer_evaluations_ + 2

    def _kernel(self, point: NDArray[np.float64]) -> FidelityQuantumKernel:
        return FidelityQuantumKernel(
            self._circuit,
            self._inputs,
            parameter_bindings={
                **self._fixed,
                **common.immutable_parameters(self._trainables, point),
            },
            sampler=self._sampler,
            shots=self._shots,
            max_statevector_qubits=self._max_statevector_qubits,
            max_statevector_bytes=self._max_statevector_bytes,
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, initial_point: Mapping[str, float]
    ) -> QuantumKernelTrainer:
        """Fit nonempty finite X and exactly two real label values.

        Initial bindings must exactly cover trainables. Invalid quantum points,
        optimizer failures and nonfinite results raise without replacing any
        previously fitted kernel. Budget termination can save a finite model
        with optimizer_success_=False. Missing SciPy raises ImportError.
        """
        data = common.data(X, len(self._inputs), nonempty=True)
        labels, classes = common.binary_labels(y, len(data))
        point = common.initial(initial_point, self._trainables)
        balanced = np.where(
            labels < 0.0,
            -1.0 / np.count_nonzero(labels < 0.0),
            1.0 / np.count_nonzero(labels > 0.0),
        )
        target = np.outer(balanced, balanced)

        def evaluate(
            candidate: NDArray[np.float64],
        ) -> tuple[FidelityQuantumKernel, NDArray[np.float64], float]:
            kernel = self._kernel(candidate)
            matrix = common.gram(kernel.evaluate(data), len(data))
            return kernel, matrix, _alignment(matrix, target)

        def objective(candidate: NDArray[np.float64]) -> float:
            return -evaluate(candidate)[2]

        initial_alignment = evaluate(point)[2]
        result = common.minimize(
            objective, point, self._max_evaluations, self._tolerance, self._initial_step
        )
        kernel, matrix, alignment = evaluate(result.point)
        self._state = _State(
            kernel,
            common.immutable_parameters(self._trainables, result.point),
            classes,
            initial_alignment,
            alignment,
            matrix,
            result,
        )
        return self


__all__ = ["QuantumKernelTrainer"]
