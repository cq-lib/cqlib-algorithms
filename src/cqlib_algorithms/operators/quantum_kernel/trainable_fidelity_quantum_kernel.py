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

"""A trainable family of Cqlib pure-state fidelity kernels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from cqlib import Circuit

from ...protocols.quantum_kernel import TrainableQuantumKernel
from ...protocols.sampler import Sampler
from .fidelity_quantum_kernel import FidelityQuantumKernel


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of names")
    result = tuple(values)
    if any(not isinstance(x, str) for x in result):
        raise TypeError(f"{label} must contain strings")
    if not result or any(not x for x in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} must contain nonempty unique names")
    return result


class TrainableFidelityQuantumKernel(TrainableQuantumKernel):
    """Snapshot a circuit with input, trainable and fixed symbols.

    The three groups are disjoint and cover all active symbols. Execution
    options follow FidelityQuantumKernel. Each complete finite real binding
    creates an independent FidelityQuantumKernel for downstream use.
    Invalid configuration raises TypeError or ValueError.
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
    ) -> None:
        self._inputs = _names(input_parameters, "input_parameters")
        self._trainables = _names(trainable_parameters, "trainable_parameters")
        if set(self._inputs) & set(self._trainables):
            raise ValueError(
                "input, trainable and fixed parameter names must be disjoint"
            )
        config = FidelityQuantumKernel(
            feature_map,
            self._inputs + self._trainables,
            parameter_bindings=parameter_bindings,
            sampler=sampler,
            shots=shots,
            max_statevector_qubits=max_statevector_qubits,
            max_statevector_bytes=max_statevector_bytes,
        )
        self._feature_map = config.feature_map
        self._fixed = dict(config.parameter_bindings)
        self._sampler = sampler
        self._shots = shots
        self._qubits = max_statevector_qubits
        self._bytes = max_statevector_bytes

    @property
    def num_features(self) -> int:
        """Return the input-column count."""
        return len(self._inputs)

    @property
    def trainable_parameters(self) -> tuple[str, ...]:
        """Return trainable names in optimization order."""
        return self._trainables

    def bind_parameters(self, values: Mapping[str, float]) -> FidelityQuantumKernel:
        """Copy exact finite bindings into a new fixed fidelity kernel."""
        if not isinstance(values, Mapping):
            raise TypeError("parameter_bindings must be a mapping")
        if set(values) != set(self._trainables):
            raise ValueError(
                "parameter_bindings must cover exactly the trainable parameters"
            )
        fixed = dict(self._fixed)
        for name in self._trainables:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("parameter_bindings values must be real numbers")
            if not math.isfinite(float(value)):
                raise ValueError("parameter_bindings values must be finite")
            fixed[name] = float(value)
        return FidelityQuantumKernel(
            self._feature_map,
            self._inputs,
            parameter_bindings=fixed,
            sampler=self._sampler,
            shots=self._shots,
            max_statevector_qubits=self._qubits,
            max_statevector_bytes=self._bytes,
        )
