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

"""Variational ground-state workflow for finite lattice systems."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import final

from cqlib.circuit import Circuit
from cqlib.qis import Hamiltonian

from ..._utils.numeric import validate_repetitions
from ...extension.lattice import (
    DoubleOccupation,
    FermiHubbardModel,
    FermionicLatticeConfiguration,
    FermionicMagnetization,
    LatticeConfiguration,
    LatticeSystem,
    SiteMagnetization,
    SiteOccupation,
    SpinCorrelation,
    SpinDownParticleNumber,
    SpinInteractionEnergy,
    SpinLatticeConfiguration,
    SpinMagnetizationSector,
    SpinUpParticleNumber,
    TotalMagnetization,
    TotalParticleNumber,
)
from ...extension.many_body import SpinAxis
from ...operators.ansatz import FermiHubbardAnsatz, SpinExchangeAnsatz
from ...protocols.ansatz import Ansatz
from ...protocols.estimator import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    require_estimator_results,
)
from ...protocols.optimizer import Optimizer
from ..variational import VQE
from .results import (
    FermionicLatticeMeasurements,
    LatticeGroundStateResult,
    SpinCorrelationMeasurement,
    SpinLatticeMeasurements,
)


def _has_nontrivial_sector(system: LatticeSystem) -> bool:
    if isinstance(system.model, FermiHubbardModel):
        sector = system.sector
        if sector is None:  # guarded by LatticeSystem, retained for typing
            return False
        dimension = math.comb(system.num_sites, sector.num_up) * math.comb(
            system.num_sites,
            sector.num_down,
        )
        return dimension > 1
    if isinstance(system.sector, SpinMagnetizationSector):
        return math.comb(system.num_sites, system.sector.num_up) > 1
    return False


def _interaction_graph_is_connected(system: LatticeSystem) -> bool:
    if system.num_sites == 1:
        return True
    adjacency: list[set[int]] = [set() for _ in range(system.num_sites)]
    for source, target, weight in system.lattice.weighted_edges:
        if source == target or weight == 0.0:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited = {0}
    pending = [0]
    while pending:
        source = pending.pop()
        for target in adjacency[source] - visited:
            visited.add(target)
            pending.append(target)
    return len(visited) == system.num_sites


def _validate_default_manifold(system: LatticeSystem) -> None:
    if _has_nontrivial_sector(system) and not _interaction_graph_is_connected(system):
        raise ValueError(
            "the default ansatz requires a connected nonzero lattice-edge graph; "
            "use a custom ansatz or solve every component-sector allocation"
        )


def _default_ansatz(
    system: LatticeSystem,
    configuration: LatticeConfiguration | None,
    *,
    reps: int,
) -> Ansatz:
    if configuration is None:
        raise ValueError("configuration is required when ansatz is not supplied")
    _validate_default_manifold(system)
    if isinstance(system.model, FermiHubbardModel):
        if not isinstance(configuration, FermionicLatticeConfiguration):
            raise TypeError(
                "a FermiHubbardModel requires a FermionicLatticeConfiguration"
            )
        return FermiHubbardAnsatz(system, configuration, reps=reps)
    if system.sector is None:
        raise ValueError(
            "a spin model without a fixed magnetization sector requires a custom ansatz"
        )
    if not isinstance(system.sector, SpinMagnetizationSector):
        raise TypeError("a spin model requires a SpinMagnetizationSector")
    if not isinstance(configuration, SpinLatticeConfiguration):
        raise TypeError("a spin model requires a SpinLatticeConfiguration")
    return SpinExchangeAnsatz(system, configuration, reps=reps)


def _fermionic_requests(
    system: LatticeSystem,
    circuit: Circuit,
    shots: int | None,
) -> tuple[list[EstimatorRequest], int]:
    model = system.model
    if not isinstance(model, FermiHubbardModel):
        raise TypeError("system must contain a FermiHubbardModel")
    observables: list[Hamiltonian] = [
        TotalParticleNumber(model).to_qubit_hamiltonian(),
        SpinUpParticleNumber(model).to_qubit_hamiltonian(),
        SpinDownParticleNumber(model).to_qubit_hamiltonian(),
        FermionicMagnetization(model).to_qubit_hamiltonian(),
    ]
    global_count = len(observables)
    observables.extend(
        SiteOccupation(model, site).to_qubit_hamiltonian()
        for site in range(system.num_sites)
    )
    observables.extend(
        DoubleOccupation(model, site).to_qubit_hamiltonian()
        for site in range(system.num_sites)
    )
    requests = [
        EstimatorRequest(
            circuit,
            observable,
            {},
            shots,
        )
        for observable in observables
    ]
    return requests, global_count


def _fermionic_measurements(
    system: LatticeSystem,
    estimates: list[EstimatorResult],
    *,
    global_count: int,
) -> FermionicLatticeMeasurements:
    sites_end = global_count + system.num_sites
    return FermionicLatticeMeasurements(
        estimates[0],
        estimates[1],
        estimates[2],
        estimates[3],
        tuple(estimates[global_count:sites_end]),
        tuple(estimates[sites_end:]),
    )


def _spin_requests(
    system: LatticeSystem,
    circuit: Circuit,
    shots: int | None,
) -> tuple[list[EstimatorRequest], tuple[tuple[int, int, SpinAxis], ...]]:
    model = system.model
    if isinstance(model, FermiHubbardModel):
        raise TypeError("system must contain a spin model")
    observables: list[Hamiltonian] = [TotalMagnetization(model).to_qubit_hamiltonian()]
    observables.extend(
        SiteMagnetization(model, site).to_qubit_hamiltonian()
        for site in range(system.num_sites)
    )
    observables.append(SpinInteractionEnergy(model).to_qubit_hamiltonian())
    axes: tuple[SpinAxis, SpinAxis, SpinAxis] = ("X", "Y", "Z")
    correlation_keys = tuple(
        (source, target, axis)
        for source, target, weight in system.lattice.weighted_edges
        if source != target and weight != 0.0
        for axis in axes
    )
    observables.extend(
        SpinCorrelation(model, source, target, axis).to_qubit_hamiltonian()
        for source, target, axis in correlation_keys
    )
    requests = [
        EstimatorRequest(
            circuit,
            observable,
            {},
            shots,
        )
        for observable in observables
    ]
    return requests, correlation_keys


def _spin_measurements(
    system: LatticeSystem,
    estimates: list[EstimatorResult],
    correlation_keys: Sequence[tuple[int, int, SpinAxis]],
) -> SpinLatticeMeasurements:
    sites_end = 1 + system.num_sites
    correlations = tuple(
        SpinCorrelationMeasurement(source, target, axis, result)
        for (source, target, axis), result in zip(
            correlation_keys,
            estimates[sites_end + 1 :],
            strict=True,
        )
    )
    return SpinLatticeMeasurements(
        estimates[0],
        tuple(estimates[1:sites_end]),
        estimates[sites_end],
        correlations,
    )


@final
class LatticeGroundState:
    """Minimize a finite lattice Hamiltonian and measure physical diagnostics.

    The built-in Fermi--Hubbard path uses the interleaved Jordan--Wigner
    occupation encoding shared by its reference state, ansatz, Hamiltonian and
    observables. Built-in spin exchange is available only for a declared,
    conserved magnetization sector. A custom ansatz may be supplied for other
    manifolds, but it remains the caller's responsibility to match the model's
    encoding and intended physical subspace.

    The returned energy is variational within the selected ansatz manifold; no
    global optimizer or exact-ground-state guarantee is implied.

    Args:
        estimator: Expectation-value estimator used by VQE and diagnostics.
        optimizer: Classical optimizer used by VQE.
        prefix: Prefix for variational parameter names.

    Raises:
        TypeError: If an injected component or ``prefix`` has the wrong type.
    """

    __slots__ = ("_estimator", "_optimizer", "_prefix")

    def __init__(
        self,
        estimator: Estimator,
        optimizer: Optimizer,
        *,
        prefix: str = "lattice",
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")
        self._estimator = estimator
        self._optimizer = optimizer
        self._prefix = prefix

    @property
    def estimator(self) -> Estimator:
        """Return the injected expectation estimator."""
        return self._estimator

    @property
    def optimizer(self) -> Optimizer:
        """Return the injected classical optimizer."""
        return self._optimizer

    @property
    def prefix(self) -> str:
        """Return the ansatz parameter prefix."""
        return self._prefix

    def run(
        self,
        system: LatticeSystem,
        initial_point: Mapping[str, float] | None = None,
        *,
        configuration: LatticeConfiguration | None = None,
        ansatz: Ansatz | None = None,
        reps: int = 1,
        shots: int | None = None,
    ) -> LatticeGroundStateResult:
        """Run VQE and evaluate the model's standard terminal observables.

        When ``ansatz`` is omitted, ``configuration`` is mandatory and a
        sector-preserving lattice ansatz is constructed. A custom ansatz is an
        explicit alternative and therefore cannot be combined with a separate
        configuration or non-default ``reps`` value.

        Args:
            system: Validated lattice model and physical sector.
            initial_point: Optional initial values keyed by circuit symbol.
            configuration: Reference occupation or spin configuration for the
                built-in sector-preserving ansatz.
            ansatz: Optional custom ansatz replacing the built-in construction.
            reps: Non-negative repetition count for the built-in ansatz.
            shots: Optional finite-shot count forwarded to the estimator.

        Returns:
            Variational ground-state result and model-specific measurements.

        Raises:
            TypeError: If an input or estimator result has the wrong type.
            ValueError: If the sector, configuration, circuit width, default
                interaction graph, or estimator result count is invalid.
        """
        if not isinstance(system, LatticeSystem):
            raise TypeError("system must be a LatticeSystem")
        validate_repetitions(reps)
        if ansatz is None:
            selected_ansatz = _default_ansatz(
                system,
                configuration,
                reps=reps,
            )
        else:
            if not isinstance(ansatz, Ansatz):
                raise TypeError("ansatz must be an Ansatz or None")
            if configuration is not None:
                raise ValueError(
                    "configuration must be omitted when a custom ansatz is supplied"
                )
            if reps != 1:
                raise ValueError("reps applies only to the built-in lattice ansatz")
            selected_ansatz = ansatz

        hamiltonian = system.model.qubit_hamiltonian()
        circuit = selected_ansatz.build_circuit(self._prefix)
        if circuit.num_qubits != hamiltonian.num_qubits:
            raise ValueError("ansatz and lattice Hamiltonian must have the same width")
        point: Mapping[str, float] = (
            dict.fromkeys(circuit.used_symbols, 0.0)
            if initial_point is None
            else initial_point
        )
        vqe_result = VQE(
            self._estimator,
            selected_ansatz,
            self._optimizer,
            prefix=self._prefix,
        ).run(
            hamiltonian,
            point,
            shots=shots,
        )

        measurements: FermionicLatticeMeasurements | SpinLatticeMeasurements
        if isinstance(system.model, FermiHubbardModel):
            requests, global_count = _fermionic_requests(
                system,
                vqe_result.optimal_circuit,
                shots,
            )
            estimates = require_estimator_results(
                self._estimator.run(requests), requests
            )
            measurements = _fermionic_measurements(
                system,
                estimates,
                global_count=global_count,
            )
        else:
            requests, correlation_keys = _spin_requests(
                system,
                vqe_result.optimal_circuit,
                shots,
            )
            estimates = require_estimator_results(
                self._estimator.run(requests), requests
            )
            measurements = _spin_measurements(
                system,
                estimates,
                correlation_keys,
            )
        return LatticeGroundStateResult(system, vqe_result, measurements)

    def compute(
        self,
        system: LatticeSystem,
        initial_point: Mapping[str, float] | None = None,
        *,
        configuration: LatticeConfiguration | None = None,
        ansatz: Ansatz | None = None,
        reps: int = 1,
        shots: int | None = None,
    ) -> LatticeGroundStateResult:
        """Solve the ground state with the same inputs and result as :meth:`run`."""
        return self.run(
            system,
            initial_point,
            configuration=configuration,
            ansatz=ansatz,
            reps=reps,
            shots=shots,
        )


__all__ = ["LatticeGroundState"]
