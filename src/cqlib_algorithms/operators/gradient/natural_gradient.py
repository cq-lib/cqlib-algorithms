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

"""Natural gradient of estimator objectives from the QGT metric.

References
----------
[1] J. Stokes, J. Izaac, N. Killoran, and G. Carleo, "Quantum natural
    gradient," Quantum, 4, 269, 2020.
    DOI: 10.22331/q-2020-05-25-269.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import final

import numpy as np
from cqlib.qis import Hamiltonian, PauliString

from ...protocols.ansatz import Ansatz
from ...protocols.estimator import Estimator
from ...protocols.gradient import GradientRequest, GradientResult
from ._linear_system import solve_metric_system
from .parameter_shift import ParameterShiftGradient
from .qgt import StatevectorQGT


@final
class NaturalGradient:
    """Natural gradient ``eta`` solving ``g @ eta = grad E`` per request.

    The metric ``g`` is the real part of the quantum geometric tensor
    (Stokes et al., arXiv:1909.02108) computed exactly by
    :class:`StatevectorQGT`, and the energy gradient is obtained from a
    single flattened :class:`ParameterShiftGradient` estimator batch.
    With ``regularization=0.0`` the component solves the (possibly
    singular) system by minimum-norm least squares; with
    ``regularization=eps > 0`` it solves the ridge system
    ``(g + eps * I) @ eta = grad E``.

    The class does not inherit from :class:`Gradient` because its output
    is a geometry-corrected direction, not the derivative of an
    expectation value.  Classical optimization loops that iterate the
    direction are out of scope.

    Raises:
        TypeError: If the estimator or a request has the wrong type.
        ValueError: If ``regularization`` is negative or non-finite, or
            a request is invalid or unsupported by the two components.
    """

    _regularization: float

    __slots__ = ("_regularization",)

    def __init__(self, regularization: float = 0.0) -> None:
        if isinstance(regularization, bool) or not isinstance(regularization, Real):
            raise TypeError("regularization must be a real number")
        value = float(regularization)
        if not math.isfinite(value):
            raise ValueError("regularization must be finite")
        if value < 0.0:
            raise ValueError("regularization must be non-negative")
        self._regularization = value

    @property
    def regularization(self) -> float:
        """Return the ridge regularization strength."""
        return float(self._regularization)

    def compute(
        self,
        estimator: Estimator,
        ansatz: Ansatz,
        observable: Hamiltonian | PauliString,
        parameter_bindings: Mapping[str, float],
        *,
        parameter_names: tuple[str, ...] | None = None,
        shots: int | None = None,
        seed: int | None = None,
        prefix: str = "theta",
    ) -> GradientResult:
        """Return one geometry-corrected direction from an ansatz objective.

        Args:
            estimator: Evaluator for parameter-shift expectation tasks.
            ansatz: Trainable component built internally.
            observable: Hermitian expectation objective.
            parameter_bindings: Exact bindings for all active symbols.
            parameter_names: Ordered subset; None uses all active symbols.
            shots: Optional expectation sample count; the metric remains exact.
            seed: Optional expectation sampler seed.
            prefix: Naming prefix used to build the ansatz.

        Returns:
            Solution of the configured metric system, in parameter order.

        Raises:
            TypeError: If component or input types are invalid.
            ValueError: If parameters or circuit support are invalid.
        """
        if not isinstance(observable, (Hamiltonian, PauliString)):
            raise TypeError("observable must be a Hamiltonian or PauliString")
        request = GradientRequest.from_ansatz(
            ansatz,
            observable,
            parameter_bindings,
            parameter_names=parameter_names,
            shots=shots,
            seed=seed,
            prefix=prefix,
        )
        return self.run(estimator, [request])[0]

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[GradientRequest],
        /,
    ) -> list[GradientResult]:
        """Return natural-gradient directions in request order."""
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        request_list = list(requests)
        if any(not isinstance(request, GradientRequest) for request in request_list):
            raise TypeError("requests must contain only GradientRequest values")
        for request in request_list:
            request.validate()

        gradients = ParameterShiftGradient().run(estimator, request_list)
        qgt = StatevectorQGT()

        results: list[GradientResult] = []
        for request, gradient in zip(request_list, gradients):
            qgt_result = qgt.compute(request)
            metric = qgt_result.metric
            values = np.asarray(gradient.values, dtype=np.float64)
            direction = solve_metric_system(metric, values, self._regularization)
            results.append(
                GradientResult(
                    request.parameter_names, tuple(float(v) for v in direction)
                )
            )
        return results
