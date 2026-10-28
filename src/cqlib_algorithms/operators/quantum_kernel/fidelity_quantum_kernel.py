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

# Modified to integrate the shared fidelity kernel into the operator layer.

"""Pure-state fidelity kernels with exact or compute-uncompute evaluation.

Reference: Havlicek et al., Nature 567, 209-212 (2019),
arXiv:1804.11326v2. The exact kernel is the Hilbert-Schmidt inner product
of the feature states' rank-one density operators.
"""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType

import numpy as np
from cqlib import Circuit, Parameter
from cqlib.qis.metrics import state_fidelity_pure
from cqlib.qis.state import Statevector
from numpy.typing import ArrayLike, NDArray

from ...protocols.sampler import Sampler
from ..state_fidelity import CircuitFidelityRequest, ComputeUncomputeFidelity
from ..state_fidelity.compute_uncompute import _validate_static_unitary
from ..state_preparation import UserCircuitPreparation
from ._validation import (
    _positive_integer,
    _real_array,
    _validate_static_unitary_circuit,
)


class FidelityQuantumKernel:
    """Evaluate ``K(x,y)=|<phi(x)|phi(y)>|**2`` using a fixed feature circuit.

    The nonempty ordered ``input_parameters`` defines data-column bindings.
    Fixed ``parameter_bindings`` must cover exactly the remaining active
    symbols. Circuits are copied and normalized by qubit insertion position.
    Both execution modes require static, invertible, unitary circuits.

    With neither ``sampler`` nor ``shots``, Core statevectors give exact local
    fidelities. The qubit and byte limits bound retained complex128 amplitude
    data; simulator scratch space and output matrices need additional memory.
    A Sampler and positive shots together select finite-shot evaluation via
    ComputeUncomputeFidelity. Shots apply to each nonredundant pair.

    Identical data rows have analytic fidelity one, including the diagonal,
    even for noisy Samplers. Self-evaluation mirrors the upper triangle.
    Returned matrices are raw fidelities without PSD correction or data
    normalization. Invalid configuration/data raises TypeError/ValueError;
    execution errors propagate. No persistent dataset cache is retained.
    """

    __slots__ = (
        "_feature_map",
        "_input_parameters",
        "_max_statevector_bytes",
        "_max_statevector_qubits",
        "_parameter_bindings",
        "_sampler",
        "_shots",
    )

    def __init__(
        self,
        feature_map: Circuit,
        input_parameters: Sequence[str],
        *,
        parameter_bindings: Mapping[str, float] | None = None,
        sampler: Sampler | None = None,
        shots: int | None = None,
        max_statevector_qubits: int = 20,
        max_statevector_bytes: int = 268_435_456,
    ) -> None:
        if isinstance(input_parameters, (str, bytes)) or not isinstance(
            input_parameters, Sequence
        ):
            raise TypeError("input_parameters must be a finite sequence of names")
        if len(input_parameters) == 0:
            raise ValueError("input_parameters must not be empty")
        names = tuple(input_parameters)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("input_parameters must contain strings")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("input_parameters must contain nonempty unique names")
        if parameter_bindings is not None and not isinstance(
            parameter_bindings, Mapping
        ):
            raise TypeError("parameter_bindings must be a mapping or None")
        fixed: dict[str, float] = {}
        for name, value in (
            {} if parameter_bindings is None else parameter_bindings
        ).items():
            if not isinstance(name, str):
                raise TypeError("parameter binding names must be strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("parameter bindings must contain real numbers")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("parameter bindings must be finite")
            fixed[name] = numeric
        if set(names) & set(fixed):
            raise ValueError("input_parameters and parameter_bindings must be disjoint")
        normalized = UserCircuitPreparation(feature_map).build_circuit("")
        if set(names) | set(fixed) != set(normalized.used_symbols):
            raise ValueError(
                "input and fixed names must cover exactly the active symbols"
            )
        if (sampler is None) != (shots is None):
            raise ValueError("sampler and shots must be supplied together")
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        if shots is not None:
            _positive_integer(shots, "shots")
        _positive_integer(max_statevector_qubits, "max_statevector_qubits")
        _positive_integer(max_statevector_bytes, "max_statevector_bytes")
        # Validate structure without selecting a point in the input domain.
        partial = normalized.assign_parameters(fixed)
        _validate_static_unitary(partial, label="feature_map")
        partial.inverse()
        constants = Circuit(partial.num_qubits)
        if not partial.global_phase.symbols:
            constants.set_global_phase(partial.global_phase)
        for operation in partial.decompose().operations:
            if operation.is_directive and operation.name == "barrier":
                constants.append(operation)
                continue
            if operation.instruction_type not in {"standard", "mcgate", "unitary"}:
                raise ValueError(
                    "feature_map must contain only static unitary operations and barriers"
                )
            if not any(
                isinstance(value, Parameter) and value.symbols
                for value in operation.params
            ):
                constants.append(operation)
        _validate_static_unitary_circuit(constants, "feature_map constant operations")
        self._feature_map = copy.deepcopy(normalized)
        self._input_parameters = names
        self._parameter_bindings = MappingProxyType(fixed)
        self._sampler = sampler
        self._shots = shots
        self._max_statevector_qubits = max_statevector_qubits
        self._max_statevector_bytes = max_statevector_bytes

    @property
    def feature_map(self) -> Circuit:
        """Return an independent normalized feature circuit."""
        return copy.deepcopy(self._feature_map)

    @property
    def input_parameters(self) -> tuple[str, ...]:
        """Return input symbol names in data-column order."""
        return self._input_parameters

    @property
    def parameter_bindings(self) -> Mapping[str, float]:
        """Return immutable fixed symbol bindings."""
        return self._parameter_bindings

    @property
    def num_features(self) -> int:
        """Return the required number of data columns."""
        return len(self._input_parameters)

    def _validate_data(self, values: ArrayLike, name: str) -> NDArray[np.float64]:
        data = _real_array(values, name, 2)
        if data.shape[1] != self.num_features:
            raise ValueError(f"{name} feature width must match input_parameters")
        return data

    def _bound(self, row: tuple[float, ...]) -> Circuit:
        bindings = {
            **self._parameter_bindings,
            **dict(zip(self._input_parameters, row)),
        }
        bound = self._feature_map.assign_parameters(bindings)
        _validate_static_unitary_circuit(bound, "bound feature_map")
        bound.inverse()
        return bound

    def evaluate(self, X: ArrayLike, Y: ArrayLike | None = None) -> NDArray[np.float64]:
        """Return a fresh ``(len(X),len(Y))`` matrix, or a symmetric Gram matrix.

        Explicit empty matrices of shape ``(0,num_features)`` are valid and
        cause no quantum execution. Complex, bool, object, ragged, nonfinite,
        and incorrectly shaped data are rejected. Resource limits are checked
        before exact simulation, and allocation failures become ValueError.
        """
        x = self._validate_data(X, "X")
        y = x if Y is None else self._validate_data(Y, "Y")
        rows, columns = len(x), len(y)
        if rows and columns > sys.maxsize // 8 // rows:
            raise ValueError(
                "kernel matrix shape exceeds representable allocation size"
            )
        try:
            matrix = np.empty((rows, columns), dtype=np.float64)
            if not rows or not columns:
                return matrix
            return self._evaluate(x, y, matrix, symmetric=Y is None)
        except MemoryError as error:
            raise ValueError("kernel evaluation allocation failed") from error

    def _evaluate(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        matrix: NDArray[np.float64],
        *,
        symmetric: bool,
    ) -> NDArray[np.float64]:
        if (
            self._sampler is None
            and self._feature_map.num_qubits > self._max_statevector_qubits
        ):
            raise ValueError("feature_map exceeds max_statevector_qubits")
        x_rows = [tuple(float(value) for value in row) for row in x]
        y_rows = (
            x_rows if symmetric else [tuple(float(value) for value in row) for row in y]
        )
        # Equal-row shortcuts still require a valid bound feature state.
        # Reuse each distinct row's validated circuit for any evaluated pairs.
        circuits = {row: self._bound(row) for row in set(x_rows) | set(y_rows)}
        pairs: list[tuple[int, int]] = []
        for i, left in enumerate(x_rows):
            for j in range(i if symmetric else 0, len(y_rows)):
                if left == y_rows[j]:
                    matrix[i, j] = 1.0
                    if symmetric:
                        matrix[j, i] = 1.0
                else:
                    pairs.append((i, j))
        if not pairs:
            return matrix
        needed_rows = {row for i, j in pairs for row in (x_rows[i], y_rows[j])}
        if self._sampler is None:
            width = self._feature_map.num_qubits
            if width > self._max_statevector_qubits:
                raise ValueError("feature_map exceeds max_statevector_qubits")
            # Test width first, then compare without allocating amplitudes.
            if 16 * (1 << width) * len(needed_rows) > self._max_statevector_bytes:
                raise ValueError("evaluation exceeds max_statevector_bytes")
        values: list[float]
        if self._sampler is None:
            states = {
                row: Statevector.from_circuit(circuit)
                for row, circuit in circuits.items()
            }
            values = []
            for i, j in pairs:
                value = float(state_fidelity_pure(states[x_rows[i]], states[y_rows[j]]))
                if not math.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
                    raise ValueError("Core fidelity must be finite and within [0, 1]")
                values.append(min(1.0, max(0.0, value)))
        else:
            assert self._shots is not None
            requests = [
                CircuitFidelityRequest(
                    circuits[x_rows[i]], circuits[y_rows[j]], self._shots
                )
                for i, j in pairs
            ]
            estimates = ComputeUncomputeFidelity(self._sampler).run(requests)
            values = [result.fidelity for result in estimates]
        for (i, j), value in zip(pairs, values):
            matrix[i, j] = value
            if symmetric:
                matrix[j, i] = value
        return matrix


__all__ = ["FidelityQuantumKernel"]
