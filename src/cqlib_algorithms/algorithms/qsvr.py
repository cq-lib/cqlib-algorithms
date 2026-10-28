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

"""Classical epsilon-SVR trained on Cqlib pure-state fidelity kernels.

The epsilon-insensitive objective follows Smola and Scholkopf,
Statistics and Computing 14, 199-222 (2004), DOI:10.1023/B:STCO.0000035301.49549.88.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..operators.quantum_kernel import FidelityQuantumKernel
from ..operators.quantum_kernel._validation import _real_array


class _SVRSolver(Protocol):
    """Only the private libsvm adapter surface used by this implementation."""

    support_: NDArray[np.intp]
    dual_coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    fit_status_: int
    n_iter_: int

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> object: ...

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class _FittedState:
    solver: _SVRSolver
    training_data: NDArray[np.float64]
    raw_kernel_matrix: NDArray[np.float64]
    kernel_matrix: NDArray[np.float64]
    correction_norm: float
    minimum_eigenvalue: float


def _real_hyperparameter(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return result


def _targets(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    result = _real_array(values, name, 1)
    if len(result) != size:
        raise ValueError(f"{name} length must match the number of samples")
    return result


def _weights(values: ArrayLike | None, size: int) -> NDArray[np.float64] | None:
    if values is None:
        return None
    result = _targets(values, size, "sample_weight")
    if np.any(result <= 0.0):
        raise ValueError("sample_weight must be strictly positive")
    return result


class QSVR:
    """Fit epsilon-SVR with a fixed FidelityQuantumKernel and libsvm.

    ``C`` and ``tol`` must be finite positive reals, ``epsilon`` finite and
    nonnegative, and ``max_iter`` -1 or a positive integer. Positive sample
    weights scale each observation's C. Scikit-learn is imported lazily by
    ``fit`` and is available through the ``machine-learning`` extra.

    By default the symmetric training Gram matrix is projected to PSD by
    clipping eigenvalues at zero. Prediction always uses raw rectangular
    fidelities; this finite training repair does not define a globally PSD
    out-of-sample kernel. Disabling repair preserves indefinite shot noise.
    A failed refit preserves the previous model. Solver convergence warnings
    propagate and an iteration-limited approximation remains fitted.
    """

    __slots__ = (
        "_C",
        "_enforce_psd",
        "_epsilon",
        "_max_iter",
        "_quantum_kernel",
        "_state",
        "_tol",
    )

    def __init__(
        self,
        quantum_kernel: FidelityQuantumKernel,
        *,
        C: float = 1.0,
        epsilon: float = 0.1,
        tol: float = 1e-3,
        max_iter: int = -1,
        enforce_psd: bool = True,
    ) -> None:
        if not isinstance(quantum_kernel, FidelityQuantumKernel):
            raise TypeError("quantum_kernel must be a FidelityQuantumKernel")
        self._C = _real_hyperparameter(C, "C")
        self._epsilon = _real_hyperparameter(epsilon, "epsilon", allow_zero=True)
        self._tol = _real_hyperparameter(tol, "tol")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer")
        if max_iter != -1 and max_iter <= 0:
            raise ValueError("max_iter must be -1 or positive")
        if not isinstance(enforce_psd, bool):
            raise TypeError("enforce_psd must be bool")
        self._quantum_kernel = quantum_kernel
        self._max_iter = max_iter
        self._enforce_psd = enforce_psd
        self._state: _FittedState | None = None

    def _fitted(self) -> _FittedState:
        if self._state is None:
            raise ValueError(
                "QSVR must be fitted before prediction or fitted diagnostics"
            )
        return self._state

    @property
    def is_fitted(self) -> bool:
        """Return whether a fit has successfully completed."""
        return self._state is not None

    @property
    def n_features_in_(self) -> int:
        """Return the feature count used at fit."""
        return int(self._fitted().training_data.shape[1])

    @property
    def training_data_(self) -> NDArray[np.float64]:
        """Return copied quantum training inputs, indexed by support_."""
        return np.array(self._fitted().training_data, dtype=np.float64, copy=True)

    @property
    def raw_kernel_matrix_(self) -> NDArray[np.float64]:
        """Return copied raw training fidelities before PSD repair."""
        return np.array(self._fitted().raw_kernel_matrix, dtype=np.float64, copy=True)

    @property
    def kernel_matrix_(self) -> NDArray[np.float64]:
        """Return a copy of the matrix passed to libsvm."""
        return np.array(self._fitted().kernel_matrix, dtype=np.float64, copy=True)

    @property
    def support_(self) -> NDArray[np.intp]:
        """Return copied support indices into training_data_."""
        return np.array(self._fitted().solver.support_, dtype=np.intp, copy=True)

    @property
    def dual_coef_(self) -> NDArray[np.float64]:
        """Return copied signed support coefficients, shape (1,n_support)."""
        return np.array(self._fitted().solver.dual_coef_, dtype=np.float64, copy=True)

    @property
    def intercept_(self) -> NDArray[np.float64]:
        """Return the copied single-output intercept array."""
        return np.array(self._fitted().solver.intercept_, dtype=np.float64, copy=True)

    @property
    def fit_status_(self) -> int:
        """Return libsvm's status: zero means no iteration-limit failure."""
        return int(self._fitted().solver.fit_status_)

    @property
    def n_iter_(self) -> int:
        """Return the single-output solver's number of iterations."""
        return int(self._fitted().solver.n_iter_)

    @property
    def psd_correction_norm_(self) -> float:
        """Return the Frobenius norm of the training PSD correction."""
        return self._fitted().correction_norm

    @property
    def minimum_raw_eigenvalue_(self) -> float:
        """Return the minimum eigenvalue of the raw symmetric training Gram."""
        return self._fitted().minimum_eigenvalue

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, sample_weight: ArrayLike | None = None
    ) -> QSVR:
        """Fit and return self, committing model state only after solver success.

        X must be nonempty real data of the kernel's feature width; y must
        be finite, one-dimensional and of matching length. Invalid inputs
        raise TypeError/ValueError before quantum work. Missing scikit-learn
        raises ImportError with installation guidance.
        """
        data = self._quantum_kernel._validate_data(X, "X")
        if not len(data):
            raise ValueError("fit requires at least one sample")
        targets = _targets(y, len(data), "y")
        weights = _weights(sample_weight, len(data))
        try:
            svm = importlib.import_module("sklearn.svm")
        except ModuleNotFoundError as error:
            if error.name not in {"sklearn", "sklearn.svm"}:
                raise
            raise ImportError(
                "QSVR.fit requires scikit-learn; install cqlib-algorithms[machine-learning]"
            ) from error
        raw = self._quantum_kernel.evaluate(data)
        if raw.shape != (len(data), len(data)) or not np.all(np.isfinite(raw)):
            raise ValueError(
                "quantum kernel must return a finite square training matrix"
            )
        if not np.array_equal(raw, raw.T):
            raise ValueError("quantum kernel must return a symmetric training matrix")
        eigenvalues, eigenvectors = np.linalg.eigh(raw)
        fitted = raw.copy()
        if self._enforce_psd:
            fitted = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            fitted = (fitted + fitted.T) / 2.0
        solver = cast(
            _SVRSolver,
            svm.SVR(
                kernel="precomputed",
                C=self._C,
                epsilon=self._epsilon,
                tol=self._tol,
                max_iter=self._max_iter,
            ),
        )
        solver.fit(fitted, targets, sample_weight=weights)
        state = _FittedState(
            solver,
            data.copy(),
            raw.copy(),
            fitted.copy(),
            float(np.linalg.norm(fitted - raw, ord="fro")),
            float(eigenvalues[0]),
        )
        self._state = state
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict with raw rectangular test/training fidelities.

        Predicting before fit raises ValueError. Empty (0,d) data returns an
        empty vector without evaluating circuits or calling the solver.
        """
        state = self._fitted()
        data = self._quantum_kernel._validate_data(X, "X")
        if not len(data):
            return np.empty(0, dtype=np.float64)
        matrix = self._quantum_kernel.evaluate(data, state.training_data)
        if matrix.shape != (len(data), len(state.training_data)) or not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                "quantum kernel must return a finite rectangular prediction matrix"
            )
        return np.array(state.solver.predict(matrix), dtype=np.float64, copy=True)

    def score(
        self, X: ArrayLike, y: ArrayLike, *, sample_weight: ArrayLike | None = None
    ) -> float:
        """Return single-output R-squared, which can be negative.

        At least two targets are required. Constant targets use finite
        scikit-learn semantics: perfect predictions score 1, otherwise 0.
        Optional weights must be finite and strictly positive.
        """
        self._fitted()
        data = self._quantum_kernel._validate_data(X, "X")
        targets = _targets(y, len(data), "y")
        if len(targets) < 2:
            raise ValueError("score requires at least two targets")
        weights = _weights(sample_weight, len(data))
        prediction = self.predict(data)
        mean = np.average(targets, weights=weights)
        numerator = float(np.average((targets - prediction) ** 2, weights=weights))
        denominator = float(np.average((targets - mean) ** 2, weights=weights))
        if numerator == 0.0:
            return 1.0
        if denominator == 0.0:
            return 0.0
        return 1.0 - numerator / denominator


__all__ = ["QSVR"]
