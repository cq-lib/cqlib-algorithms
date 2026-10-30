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

"""Fixed and trainable quantum kernels and scalar kernel losses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class QuantumKernel(ABC):
    """A fixed kernel with a positive data-column count.

    Self-evaluation returns a finite real symmetric Gram matrix. Cross
    evaluation returns shape (len(X), len(Y)). Implementations define the
    input domain and return independent arrays.
    """

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Return the data-column count."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X: ArrayLike, Y: ArrayLike | None = None) -> NDArray[np.float64]:
        """Evaluate a Gram or cross-kernel matrix."""
        raise NotImplementedError


class TrainableQuantumKernel(ABC):
    """A kernel family with explicitly ordered finite real parameters.

    Binding copies a complete mapping into an independent fixed kernel.
    Later bindings preserve previously returned kernel configurations.
    Execution services such as a Sampler may be shared.
    """

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Return the positive data-column count."""
        raise NotImplementedError

    @property
    @abstractmethod
    def trainable_parameters(self) -> tuple[str, ...]:
        """Return nonempty unique names in optimization order."""
        raise NotImplementedError

    @abstractmethod
    def bind_parameters(self, values: Mapping[str, float]) -> QuantumKernel:
        """Copy exact finite bindings into a new fixed kernel."""
        raise NotImplementedError

    def evaluate(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
        *,
        parameter_bindings: Mapping[str, float],
    ) -> NDArray[np.float64]:
        """Bind supplied parameters and evaluate the resulting kernel."""
        return self.bind_parameters(parameter_bindings).evaluate(X, Y)


class KernelLoss(Protocol):
    """Evaluate a finite scalar loss from a Gram matrix and matching targets.

    The trainer supplies independent finite real arrays. The loss defines
    target semantics, including class counts or regression target ranges.
    """

    def __call__(
        self, matrix: NDArray[np.float64], targets: NDArray[np.float64], /
    ) -> float: ...
