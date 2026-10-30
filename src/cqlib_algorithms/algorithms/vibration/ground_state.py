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

"""Variational ground-state workflow for finite-modal vibrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import final

from cqlib import Circuit
from cqlib.qis import Hamiltonian

from ...extension.vibration import ModalOccupation, VibrationalSystem
from ...operators.ansatz import UVCCSDAnsatz
from ...operators.state_preparation import VSCFStatePreparation
from ...protocols.ansatz import Ansatz
from ...protocols.estimator import Estimator, EstimatorRequest, EstimatorResult
from ...protocols.optimizer import Optimizer
from ..variational import VQE
from .results import VibrationalGroundStateResult, VibrationalMeasurements


class _FixedVSCFAnsatz(Ansatz):
    """Adapt a zero-parameter VSCF reference to the nominal Ansatz contract."""

    __slots__ = ("_preparation",)

    def __init__(self, preparation: VSCFStatePreparation) -> None:
        self._preparation = preparation
        self.validate()

    def validate(self) -> None:
        """Validate the delegated fixed state preparation."""
        self._preparation.validate()

    def build_circuit(self, prefix: str) -> Circuit:
        """Return an independent fixed reference circuit."""
        return self._preparation.build_circuit(prefix)

    def num_parameters(self) -> int:
        """Return zero because the VSCF reference is fixed."""
        return 0

    def num_qubits(self) -> int:
        """Return the one-hot modal-register width."""
        return self._preparation.num_qubits()


def _validate_repetitions(reps: int) -> None:
    """Validate a non-negative repetition count."""
    if type(reps) is not int:
        raise TypeError("reps must be an int")
    if reps < 0:
        raise ValueError("reps must be non-negative")


def _normalize_estimates(
    estimates: object,
    *,
    expected: int,
) -> list[EstimatorResult]:
    """Validate and return estimator results in their original order."""
    if not isinstance(estimates, list):
        raise TypeError("estimator must return a list of EstimatorResult values")
    if len(estimates) != expected:
        raise ValueError(
            "estimator must return one result per auxiliary-observable request"
        )
    if any(not isinstance(result, EstimatorResult) for result in estimates):
        raise TypeError("estimator must return EstimatorResult values")
    return estimates


def _auxiliary_observables(system: VibrationalSystem) -> tuple[Hamiltonian, ...]:
    space = system.modal_space
    modal_occupations = tuple(
        ModalOccupation(space, mode, modal).to_qubit_hamiltonian()
        for mode, count in enumerate(space.num_modals)
        for modal in range(count)
    )
    return (
        *modal_occupations,
        *system.occupied_modals.to_qubit_hamiltonians(),
        *system.one_hot_leakage.to_qubit_hamiltonians(),
    )


def _measurements(
    system: VibrationalSystem,
    estimates: Sequence[EstimatorResult],
) -> VibrationalMeasurements:
    rows: list[tuple[EstimatorResult, ...]] = []
    offset = 0
    for count in system.modal_space.num_modals:
        rows.append(tuple(estimates[offset : offset + count]))
        offset += count
    mode_end = offset + system.num_modes
    return VibrationalMeasurements(
        tuple(rows),
        tuple(estimates[offset:mode_end]),
        tuple(estimates[mode_end:]),
    )


@final
class VibrationalGroundState:
    """Minimize a finite-modal vibrational Hamiltonian and diagnose its sector.

    The built-in path prepares the selected VSCF product reference and applies
    an ordered UVCCSD factorization. Optimization is delegated unchanged to
    :class:`VQE`. The final circuit is measured once as a batch for every
    modal occupation, per-mode total occupation, and strict one-hot leakage.

    A custom ansatz is accepted when it has the mapped Hamiltonian width. Its
    compatibility with the one-hot physical sector is the caller's
    responsibility; the returned squared one-hot constraints provide strict
    leakage diagnostics. The returned energy is variational within the
    selected ansatz manifold and does not imply global optimizer convergence
    or an exact ground-state guarantee.

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
        prefix: str = "uvcc",
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

    def compute(
        self,
        system: VibrationalSystem,
        initial_point: Mapping[str, float] | None = None,
        *,
        reference_modals: Sequence[int] | None = None,
        ansatz: Ansatz | None = None,
        reps: int = 1,
        shots: int | None = None,
    ) -> VibrationalGroundStateResult:
        """Run VQE and evaluate modal-occupation diagnostics.

        With no custom ``ansatz``, ``reference_modals`` selects the modal in
        each mode occupied by the fixed VSCF product reference; omitting it
        selects modal zero in every mode. ``reps`` controls independent UVCCSD
        layers. A custom ansatz is an explicit alternative and therefore
        cannot be combined with ``reference_modals`` or non-default ``reps``.

        When ``initial_point`` is omitted, every symbol actually used by the
        selected circuit is initialized to zero. On the built-in path this
        makes the initial variational state exactly the selected VSCF product
        reference; it does not solve separate self-consistent modal equations.

        Args:
            system: Vibrational system in a finite modal basis.
            initial_point: Optional initial values keyed by circuit symbol.
            reference_modals: Optional occupied modal for each vibrational mode.
            ansatz: Optional custom ansatz replacing UVCCSD.
            reps: Non-negative repetition count for the built-in ansatz.
            shots: Optional finite-shot count forwarded to the estimator.

        Returns:
            Variational ground-state result and one-hot sector diagnostics.

        Raises:
            TypeError: If an input or estimator result has the wrong type.
            ValueError: If selections, circuit width, or result counts disagree.
        """
        if not isinstance(system, VibrationalSystem):
            raise TypeError("system must be a VibrationalSystem")
        _validate_repetitions(reps)
        selected_ansatz: Ansatz
        if ansatz is None:
            if all(count == 1 for count in system.modal_space.num_modals):
                selected_ansatz = _FixedVSCFAnsatz(
                    VSCFStatePreparation(system.modal_space, reference_modals)
                )
            else:
                selected_ansatz = UVCCSDAnsatz(
                    system.modal_space,
                    reference_modals=reference_modals,
                    reps=reps,
                )
        else:
            if not isinstance(ansatz, Ansatz):
                raise TypeError("ansatz must be an Ansatz or None")
            if reference_modals is not None:
                raise ValueError(
                    "reference_modals must be omitted when a custom ansatz is supplied"
                )
            if reps != 1:
                raise ValueError("reps applies only to the built-in UVCCSD ansatz")
            selected_ansatz = ansatz

        hamiltonian = system.to_qubit_hamiltonian()
        circuit = selected_ansatz.build_circuit(self._prefix)
        if circuit.num_qubits != hamiltonian.num_qubits:
            raise ValueError(
                "ansatz and vibrational Hamiltonian must have the same width"
            )
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
        ).compute_minimum_eigenvalue(
            hamiltonian,
            point,
            shots=shots,
        )

        observables = _auxiliary_observables(system)
        requests = [
            EstimatorRequest(
                vqe_result.optimal_circuit,
                observable,
                {},
                shots,
            )
            for observable in observables
        ]
        estimates = _normalize_estimates(
            self._estimator.run(requests),
            expected=len(requests),
        )
        return VibrationalGroundStateResult(
            system,
            vqe_result,
            _measurements(system, estimates),
        )


__all__ = ["VibrationalGroundState"]
