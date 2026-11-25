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

"""Electronic ground-state workflow composed from reusable public components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

from cqlib.qis import FermionToQubitMapper, JordanWignerMapper

from ...extension.electronic_structure.electronic import (
    AngularMomentum,
    ElectronicEnergy,
    ElectronicSystem,
    Magnetization,
    ParticleNumber,
)
from ...operators.ansatz.uccsd_ansatz import UCCSDAnsatz
from ...protocols.ansatz import Ansatz
from ...protocols.estimator import (
    Estimator,
    EstimatorRequest,
    require_estimator_results,
)
from ...protocols.optimizer import Optimizer
from ..variational.vqe import VQE
from .results import ElectronicGroundStateResult


@final
class ElectronicGroundState:
    """Minimize an electronic Hamiltonian and evaluate symmetry observables.

    The default circuit is the existing ordered UCCSD ansatz with its
    Hartree--Fock reference. Optimization is delegated unchanged to VQE; this
    workflow only assembles the chemistry inputs and performs one batched
    terminal evaluation of particle number, magnetization and total spin.

    Args:
        estimator: Expectation-value estimator used by VQE and diagnostics.
        optimizer: Classical optimizer used by VQE.
        mapper: Fermion-to-qubit mapper shared by all mapped operators. The
            Jordan--Wigner mapping is used when omitted.
        prefix: Prefix for variational parameter names.

    Raises:
        TypeError: If an injected component or ``prefix`` has the wrong type.
    """

    __slots__ = ("_estimator", "_mapper", "_optimizer", "_prefix")

    def __init__(
        self,
        estimator: Estimator,
        optimizer: Optimizer,
        *,
        mapper: FermionToQubitMapper | None = None,
        prefix: str = "ucc",
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        selected_mapper = JordanWignerMapper() if mapper is None else mapper
        if not isinstance(selected_mapper, FermionToQubitMapper):
            raise TypeError("mapper must implement FermionToQubitMapper")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a str")
        self._estimator = estimator
        self._optimizer = optimizer
        self._mapper = selected_mapper
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
    def mapper(self) -> FermionToQubitMapper:
        """Return the fermion-to-qubit mapper shared by all operators."""
        return self._mapper

    @property
    def prefix(self) -> str:
        """Return the ansatz parameter prefix."""
        return self._prefix

    def run(
        self,
        system: ElectronicSystem,
        initial_point: Mapping[str, float] | None = None,
        *,
        ansatz: Ansatz | None = None,
        shots: int | None = None,
    ) -> ElectronicGroundStateResult:
        """Run VQE and evaluate particle-number and spin diagnostics.

        A custom ansatz must use the mapped Hamiltonian width. When no initial
        point is supplied, every active parameter of the selected circuit is
        initialized to zero.

        Args:
            system: Electronic system containing integrals and occupations.
            initial_point: Optional initial values keyed by circuit symbol.
            ansatz: Optional replacement for the default UCCSD ansatz.
            shots: Optional finite-shot count forwarded to the estimator.

        Returns:
            Ground-state energy, optimized circuit, and symmetry diagnostics.

        Raises:
            TypeError: If an input or estimator result has the wrong type.
            ValueError: If circuit widths or estimator result counts disagree.
        """
        if not isinstance(system, ElectronicSystem):
            raise TypeError("system must be an ElectronicSystem")
        selected_ansatz = (
            UCCSDAnsatz(system.occupation, mapper=self._mapper)
            if ansatz is None
            else ansatz
        )
        if not isinstance(selected_ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz or None")

        hamiltonian = ElectronicEnergy(system.integrals).to_qubit_hamiltonian(
            self._mapper
        )
        circuit = selected_ansatz.build_circuit(self._prefix)
        if circuit.num_qubits != hamiltonian.num_qubits:
            raise ValueError("ansatz and mapped Hamiltonian must have the same width")
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

        factories = (
            ("particle_number", ParticleNumber(system.num_spatial_orbitals)),
            ("magnetization", Magnetization(system.num_spatial_orbitals)),
            (
                "angular_momentum",
                AngularMomentum(
                    system.num_spatial_orbitals,
                    system.alpha_beta_overlap,
                ),
            ),
        )
        requests = [
            EstimatorRequest(
                vqe_result.optimal_circuit,
                factory.to_qubit_hamiltonian(self._mapper),
                {},
                shots,
            )
            for _, factory in factories
        ]
        estimates = require_estimator_results(self._estimator.run(requests), requests)
        auxiliary = {
            name: result for (name, _), result in zip(factories, estimates, strict=True)
        }
        return ElectronicGroundStateResult(
            system,
            vqe_result,
            auxiliary,
            mapper=self._mapper,
        )

    def compute(
        self,
        system: ElectronicSystem,
        initial_point: Mapping[str, float] | None = None,
        *,
        ansatz: Ansatz | None = None,
        shots: int | None = None,
    ) -> ElectronicGroundStateResult:
        """Solve the ground state with the same inputs and result as :meth:`run`."""
        return self.run(system, initial_point, ansatz=ansatz, shots=shots)


__all__ = ["ElectronicGroundState"]
