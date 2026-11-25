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

"""Abstract gradient interface and value types for estimator objectives."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

from cqlib.qis import Hamiltonian, PauliString

from .ansatz import Ansatz
from .estimator import Estimator, EstimatorRequest


@dataclass(frozen=True, slots=True)
class GradientRequest:
    """A gradient request for one estimator expectation objective.

    The scalar objective is the ``expectation`` returned for
    ``estimator_request``. ``parameter_names`` is an explicit ordered,
    duplicate-free subset of that request's bindings. An empty tuple requests
    an empty gradient.

    The dataclass is only shallowly frozen because its estimator request holds
    mutable Core objects. Callers must not modify those objects during the
    request's lifetime. A gradient implementation must nevertheless call
    :meth:`validate` immediately before constructing perturbed requests.
    Revalidation detects invalid mutations but cannot detect every semantically
    different mutation that remains valid.

    Args:
        estimator_request: Fully specified expectation-value objective.
        parameter_names: Ordered subset of bound parameters to differentiate.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If parameter names are duplicated or unbound, or if the
            nested estimator request is no longer valid.
        cqlib.circuit.CircuitError: If nested ``Circuit.validate`` fails.
    """

    estimator_request: EstimatorRequest
    parameter_names: tuple[str, ...]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_ansatz(
        cls,
        ansatz: Ansatz,
        observable: Hamiltonian | PauliString | None,
        parameter_bindings: Mapping[str, float],
        *,
        parameter_names: tuple[str, ...] | None = None,
        shots: int | None = None,
        seed: int | None = None,
        prefix: str = "theta",
    ) -> GradientRequest:
        """Build an independent objective request from a trainable component.

        Args:
            ansatz: Trainable component built once using ``prefix``.
            observable: Hermitian objective; ``None`` supplies identity for geometry.
            parameter_bindings: Exact finite bindings for all active symbols.
            parameter_names: Ordered derivative subset; None uses active-symbol order.
            shots: Optional expectation sample count.
            seed: Optional expectation sampler seed.
            prefix: Naming prefix for the component's newly created symbols.

        Returns:
            A request owning independent circuit and observable snapshots.

        Raises:
            TypeError: If component, bindings, or configuration types are invalid.
            ValueError: If component metadata, symbols, or dimensions disagree.
        """
        # Import at construction time so protocol initialization stays independent
        # of the shared builder, which also serves preparation components.
        from .._utils.state_preparation import build_component_circuit

        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz")
        circuit = deepcopy(build_component_circuit(ansatz, prefix=prefix))
        if observable is None:
            observable = PauliString.from_str("I" * circuit.num_qubits)
        if not isinstance(observable, (Hamiltonian, PauliString)):
            raise TypeError("observable must be a Hamiltonian, PauliString, or None")
        return cls(
            EstimatorRequest(
                circuit, observable.copy(), parameter_bindings, shots, seed
            ),
            tuple(circuit.used_symbols) if parameter_names is None else parameter_names,
        )

    def validate(self) -> None:
        """Revalidate this request and its nested estimator request.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If names are duplicated or unbound, or if the nested
                estimator request violates an invariant.
            cqlib.circuit.CircuitError: If nested ``Circuit.validate`` fails.
        """
        if not isinstance(self.estimator_request, EstimatorRequest):
            raise TypeError("estimator_request must be an EstimatorRequest")
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")

        self.estimator_request.validate()
        bound_names = set(self.estimator_request.parameter_bindings)
        unknown = [name for name in self.parameter_names if name not in bound_names]
        if unknown:
            raise ValueError(f"gradient parameters must be bound: {unknown!r}")


@dataclass(frozen=True, slots=True)
class GradientResult:
    """Finite gradient values paired with their parameter names.

    Attributes:
        parameter_names: Differentiated names in result order.
        values: Derivatives in the same order.

    Raises:
        TypeError: If fields are not tuples or contain values of the wrong type.
        ValueError: If names are duplicated, lengths differ, or a value is not finite.
    """

    parameter_names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple")
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter names must be unique")
        if len(self.parameter_names) != len(self.values):
            raise ValueError(
                "parameter names and gradient values must have equal length"
            )

        normalized: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("gradient values must be real numbers")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("gradient values must be finite")
            normalized.append(numeric_value)
        object.__setattr__(self, "values", tuple(normalized))


class Gradient(ABC):
    """Abstract base class for synchronous batched estimator gradients.

    Implementations differentiate only ``EstimatorResult.expectation``. They
    call :meth:`GradientRequest.validate` before evaluation, should flatten all
    perturbed points into as few estimator batches as possible, and return one
    result per request in input order. The estimator is explicit so the
    abstract base class does not impose a constructor on implementations.
    """

    __slots__ = ()

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
        """Differentiate one ansatz objective using this component's batch method.

        Args:
            estimator: Evaluator for the generated expectation tasks.
            ansatz: Trainable component to build internally.
            observable: Hermitian objective on the ansatz register.
            parameter_bindings: Exact bindings for the built circuit's symbols.
            parameter_names: Ordered subset; None uses all active symbols.
            shots: Optional expectation sample count.
            seed: Optional expectation sampler seed.
            prefix: Naming prefix used to build the ansatz.

        Returns:
            One gradient result in the selected parameter order.

        Raises:
            TypeError: If a component, input, or response has the wrong type.
            ValueError: If inputs or the response violate their contracts.
        """
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
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
        results = self.run(estimator, [request])
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError("gradient must return one result per request")
        result = results[0]
        if not isinstance(result, GradientResult):
            raise TypeError("gradient must return GradientResult values")
        if result.parameter_names != request.parameter_names:
            raise ValueError("gradient result parameter order must match the request")
        return result

    @abstractmethod
    def run(
        self,
        estimator: Estimator,
        requests: Sequence[GradientRequest],
        /,
    ) -> list[GradientResult]:
        """Differentiate a batch of estimator expectation objectives.

        Args:
            estimator: Expectation estimator for shifted objectives.
            requests: Gradient requests in output order.

        Returns:
            One gradient result per request.
        """
        raise NotImplementedError
