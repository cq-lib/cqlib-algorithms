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

"""Hamiltonian phase estimation through scaled real-time evolution.

References
----------
[1] A. Aspuru-Guzik et al., "Simulated quantum computation of molecular
    energies," Science, 309, 1704-1707, 2005.
    DOI: 10.1126/science.1113479.
[2] R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca, "Quantum algorithms
    revisited," Proceedings of the Royal Society A, 454, 339-354, 1998.
    DOI: 10.1098/rspa.1998.0164.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import final

from cqlib import Circuit
from cqlib.qis import Hamiltonian, TrotterMode

from ..._utils.hamiltonian import (
    snapshot_nonzero_hermitian_hamiltonian,
)
from ..._utils.numeric import normalize_positive_integer
from ..._utils.phase_estimation import (
    normalize_evaluation_qubits,
    normalize_shots,
    snapshot_static_unitary_circuit,
)
from ...protocols import Sampler
from .problem import PhaseEstimationProblem
from .standard import (
    PhaseEstimation,
    PhaseEstimationRequest,
    PhaseEstimationResult,
)


def _coefficient_bound(hamiltonian: Hamiltonian) -> float:
    try:
        bound = math.fsum(
            abs(complex(coefficient) * pauli.phase.to_complex())
            for pauli, coefficient in hamiltonian.terms
        )
    except OverflowError as error:
        raise ValueError("hamiltonian coefficient bound must be finite") from error
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("hamiltonian must have a positive finite coefficient bound")
    return bound


def _resolved_bound(hamiltonian: Hamiltonian, bound: float | None) -> float:
    if bound is None:
        return _coefficient_bound(hamiltonian)
    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
        raise TypeError("bound must be a real number or None")
    resolved = float(bound)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("bound must be positive and finite")
    return resolved


@dataclass(frozen=True, slots=True)
class HamiltonianPhaseEstimationRequest:
    """One Hamiltonian phase-estimation request.

    ``bound`` is an upper bound on the absolute eigenvalues. If omitted, the
    sum of absolute Pauli coefficients is used. Non-commuting terms use the
    selected Core product formula and ``trotter_steps``.

    Args:
        hamiltonian: Finite Hermitian Pauli Hamiltonian.
        state_preparation: Fully bound state-preparation circuit.
        num_evaluation_qubits: Width of the QPE phase register.
        shots: Positive number of circuit samples.
        bound: Positive spectral bound, or ``None`` to use a coefficient bound.
        trotter_steps: Positive product-formula step count.
        trotter_mode: Cqlib product formula, or ``None`` for second order.
        approximation_degree: Small-angle inverse-QFT layers to omit.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If dimensions, bounds, or numeric settings are invalid.
    """

    hamiltonian: Hamiltonian
    state_preparation: Circuit
    num_evaluation_qubits: int
    shots: int
    bound: float | None = None
    trotter_steps: int = 1
    trotter_mode: TrotterMode | None = None
    approximation_degree: int = 0

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a Hamiltonian")
        hamiltonian = snapshot_nonzero_hermitian_hamiltonian(
            self.hamiltonian,
            name="hamiltonian",
        )
        state_preparation = snapshot_static_unitary_circuit(
            self.state_preparation,
            "state_preparation",
        )
        if state_preparation.num_qubits != hamiltonian.num_qubits:
            raise ValueError(
                "state_preparation and hamiltonian must have the same qubit count"
            )
        object.__setattr__(self, "hamiltonian", hamiltonian)
        object.__setattr__(self, "state_preparation", state_preparation)
        if self.trotter_mode is None:
            object.__setattr__(self, "trotter_mode", TrotterMode.second_order())
        elif not isinstance(self.trotter_mode, TrotterMode):
            raise TypeError("trotter_mode must be a TrotterMode or None")
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallowly frozen request before execution."""
        snapshot_nonzero_hermitian_hamiltonian(self.hamiltonian, name="hamiltonian")
        snapshot_static_unitary_circuit(
            self.state_preparation,
            "state_preparation",
        )
        if self.state_preparation.num_qubits != self.hamiltonian.num_qubits:
            raise ValueError(
                "state_preparation and hamiltonian must have the same qubit count"
            )
        width = normalize_evaluation_qubits(self.num_evaluation_qubits)
        normalize_shots(self.shots)
        _resolved_bound(self.hamiltonian, self.bound)
        normalize_positive_integer(self.trotter_steps, "trotter_steps")
        if not isinstance(self.trotter_mode, TrotterMode):
            raise TypeError("trotter_mode must be a TrotterMode")
        if isinstance(self.approximation_degree, bool) or not isinstance(
            self.approximation_degree, int
        ):
            raise TypeError("approximation_degree must be an integer")
        if not 0 <= self.approximation_degree < width:
            raise ValueError(
                "approximation_degree must satisfy "
                "0 <= approximation_degree < num_evaluation_qubits"
            )


@dataclass(frozen=True, slots=True)
class HamiltonianPhaseEstimationResult:
    """Store an eigenvalue estimate recovered from a QPE phase distribution.

    Attributes:
        eigenvalue: Selected signed eigenvalue estimate.
        eigenvalue_probabilities: Mapped empirical eigenvalue distribution.
        bound: Spectral bound used for scaling.
        scale: Evolution scale ``pi / bound``.
        phase_result: Underlying standard-QPE result.
    """

    eigenvalue: float
    eigenvalue_probabilities: Mapping[float, float]
    bound: float
    scale: float
    phase_result: PhaseEstimationResult

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name, value in (
            ("eigenvalue", self.eigenvalue),
            ("bound", self.bound),
            ("scale", self.scale),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite real number")
        if self.bound <= 0.0 or self.scale <= 0.0:
            raise ValueError("bound and scale must be positive")
        if not isinstance(self.phase_result, PhaseEstimationResult):
            raise TypeError("phase_result must be a PhaseEstimationResult")
        if not isinstance(self.eigenvalue_probabilities, Mapping):
            raise TypeError("eigenvalue_probabilities must be a mapping")
        probabilities = dict(self.eigenvalue_probabilities)
        if not probabilities or not math.isclose(
            sum(float(value) for value in probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("eigenvalue probabilities must sum to one")
        object.__setattr__(self, "eigenvalue", float(self.eigenvalue))
        object.__setattr__(self, "bound", float(self.bound))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(
            self,
            "eigenvalue_probabilities",
            MappingProxyType(
                {
                    float(value): float(probabilities[value])
                    for value in sorted(probabilities)
                }
            ),
        )


def _phase_to_eigenvalue(phase: float, scale: float) -> float:
    signed_phase = phase if phase < 0.5 else phase - 1.0
    return 2.0 * math.pi * signed_phase / scale


@final
class HamiltonianPhaseEstimation:
    """Estimate eigenvalues by applying QPE to ``exp(i * scale * H)``.

    Core synthesizes ``exp(-i*t*H)``, so the implementation uses negative
    ``t``. The scale ``pi / bound`` maps eigenvalues from ``[-bound, bound]``
    to normalized phases in ``[-1/2, 1/2]``. The endpoints alias at phase
    ``1/2``; callers that may occupy an endpoint should provide a strictly
    larger bound. For non-commuting Pauli terms the returned distribution is
    that of the selected product-formula approximation.

    Args:
        sampler: Sampling implementation used by the internal QPE algorithm.

    Raises:
        TypeError: If the sampler or a request has the wrong type.
        ValueError: If Hamiltonian, evolution, or QPE validation fails.
    """

    __slots__ = ("_phase_estimation",)

    def __init__(self, sampler: Sampler) -> None:
        self._phase_estimation = PhaseEstimation(sampler)

    @property
    def sampler(self) -> Sampler:
        """Return the injected Sampler implementation."""
        return self._phase_estimation.sampler

    @staticmethod
    def build_unitary(
        hamiltonian: Hamiltonian,
        *,
        bound: float | None = None,
        trotter_steps: int = 1,
        trotter_mode: TrotterMode | None = None,
    ) -> tuple[Circuit, float, float]:
        """Build ``exp(i*scale*H)`` and return circuit, bound, and scale."""
        snapshot = snapshot_nonzero_hermitian_hamiltonian(
            hamiltonian, name="hamiltonian"
        )
        resolved_bound = _resolved_bound(snapshot, bound)
        normalize_positive_integer(trotter_steps, "trotter_steps")
        mode = TrotterMode.second_order() if trotter_mode is None else trotter_mode
        if not isinstance(mode, TrotterMode):
            raise TypeError("trotter_mode must be a TrotterMode or None")
        scale = math.pi / resolved_bound
        circuit = snapshot.to_evolution_circuit(
            -scale,
            trotter_steps,
            mode,
        )
        return circuit, resolved_bound, scale

    def run(
        self,
        requests: Sequence[HamiltonianPhaseEstimationRequest],
        /,
    ) -> list[HamiltonianPhaseEstimationResult]:
        """Execute Hamiltonian phase-estimation requests as one QPE batch."""
        phase_requests: list[PhaseEstimationRequest] = []
        resolved: list[tuple[float, float]] = []
        for request in requests:
            if not isinstance(request, HamiltonianPhaseEstimationRequest):
                raise TypeError(
                    "requests must contain HamiltonianPhaseEstimationRequest instances"
                )
            request.validate()
            unitary, bound, scale = self.build_unitary(
                request.hamiltonian,
                bound=request.bound,
                trotter_steps=request.trotter_steps,
                trotter_mode=request.trotter_mode,
            )
            phase_requests.append(
                PhaseEstimationRequest(
                    PhaseEstimationProblem(unitary, request.state_preparation),
                    request.num_evaluation_qubits,
                    request.shots,
                    request.approximation_degree,
                )
            )
            resolved.append((bound, scale))

        phase_results = self._phase_estimation.run(phase_requests)
        results: list[HamiltonianPhaseEstimationResult] = []
        for phase_result, (bound, scale) in zip(phase_results, resolved):
            probabilities: dict[float, float] = {}
            dimension = 1 << phase_result.num_evaluation_qubits
            for point, probability in phase_result.phase_probabilities.items():
                eigenvalue = _phase_to_eigenvalue(point / dimension, scale)
                probabilities[eigenvalue] = (
                    probabilities.get(eigenvalue, 0.0) + probability
                )
            results.append(
                HamiltonianPhaseEstimationResult(
                    eigenvalue=_phase_to_eigenvalue(phase_result.phase, scale),
                    eigenvalue_probabilities=probabilities,
                    bound=bound,
                    scale=scale,
                    phase_result=phase_result,
                )
            )
        return results


__all__ = [
    "HamiltonianPhaseEstimation",
    "HamiltonianPhaseEstimationRequest",
    "HamiltonianPhaseEstimationResult",
]
