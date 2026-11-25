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

"""Qubit-wise Pauli measurement plans and sampler-ready basis circuits.

References
----------
[1] O. Crawford et al., "Efficient quantum measurement of Pauli operators in
    the presence of finite sampling error," Quantum, 5, 385, 2021.
    DOI: 10.22331/q-2021-01-20-385.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import final

from cqlib import Circuit, Qubit
from cqlib.circuit import CircuitError
from cqlib.device import ExecutionResult
from cqlib.qis import Pauli, PauliString

from ..._utils.parameters import bind_circuit_parameters, validate_parameter_prefix
from ..._utils.state_preparation import build_component_circuit
from ...protocols.ansatz import Ansatz
from ...protocols.sampler import Sampler, SamplerRequest, require_sampler_results
from ...protocols.state_preparation import StatePreparation
from .pauli_grouping_strategy import PauliGroupingStrategy, snapshot_terms


def _validate_preparation_circuit(circuit: Circuit) -> None:
    """Reject prior measurements, resets and dynamic execution, even nested."""
    circuit.validate()
    if circuit.num_qubits <= 0:
        raise ValueError("circuit must have at least one qubit")
    try:
        expanded = circuit.decompose()
    except CircuitError as exc:
        raise ValueError("circuit must be a decomposable quantum preparation") from exc
    for operation in expanded.operations:
        if operation.instruction_type in ("standard", "mcgate", "unitary", "delay"):
            continue
        if (
            operation.instruction_type == "directive"
            and operation.name.lower() == "barrier"
        ):
            continue
        raise ValueError(
            "circuit must not contain measurement, reset or dynamic operations"
        )


@dataclass(frozen=True, slots=True)
class MeasurementGroup:
    """Original term positions and local basis, indexed by circuit insertion order.

    A basis entry is ``I``, ``X``, ``Y`` or ``Z``. ``I`` imposes no basis
    rotation; all physical qubits remain part of the group's sampler request.

    Attributes:
        term_indices: Original observable positions assigned to the group.
        basis: Local product-Pauli measurement basis in circuit-qubit order.
    """

    term_indices: tuple[int, ...]
    basis: tuple[str, ...]


@final
class MeasurementPlan:
    """Private snapshots of a preparation, Pauli terms and QWC execution groups.

    ``term(index)`` returns a copy retaining the original scalar phase.
    Identity-only terms are listed in ``identity_indices`` and require no
    execution group; their signed or complex phase must be handled by a later
    coefficient-aware estimator. ``build_circuit`` returns a fresh, unmeasured
    basis circuit on every call and never exposes or mutates the stored
    preparation. A Sampler owns the terminal measurement operation.

    Args:
        circuit: Preparation snapshot.
        terms: Observable terms in original order.
        groups: QWC groups over original term positions.
        identity_indices: Identity-only term positions.

    Raises:
        TypeError: If stored values have invalid types.
        ValueError: If dimensions, indices, or group bases are inconsistent.
    """

    __slots__ = ("_circuit", "_groups", "_identity_indices", "_qubits", "_terms")

    def __init__(
        self,
        circuit: Circuit,
        terms: tuple[PauliString, ...],
        groups: tuple[MeasurementGroup, ...],
        identity_indices: tuple[int, ...],
    ) -> None:
        self._circuit = copy.copy(circuit)
        self._terms = tuple(term.copy() for term in terms)
        self._qubits = tuple(self._circuit.qubits)
        self._groups = groups
        self._identity_indices = identity_indices

    @property
    def groups(self) -> tuple[MeasurementGroup, ...]:
        """Return immutable group metadata in execution order."""
        return self._groups

    @property
    def identity_indices(self) -> tuple[int, ...]:
        """Return original positions of scalar identity Pauli terms."""
        return self._identity_indices

    @property
    def qubits(self) -> tuple[Qubit, ...]:
        """Return measurement qubits in insertion order."""
        return self._qubits

    @property
    def num_terms(self) -> int:
        """Return the number of input terms, including identities."""
        return len(self._terms)

    def term(self, index: int) -> PauliString:
        """Return a copy of one term at a non-negative, zero-based index.

        Boolean and non-integer indices raise ``TypeError``; negative and
        out-of-range indices raise ``IndexError``.
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= len(self._terms):
            raise IndexError("index is out of range")
        return self._terms[index].copy()

    def build_circuit(self, group_index: int) -> Circuit:
        """Return a private, unmeasured basis circuit for a QWC group.

        The local basis maps Pauli X to computational Z via H and Pauli Y
        via Sdg then H. The returned circuit is suitable for ``SamplerRequest``;
        the Sampler adds terminal measurements in ``plan.qubits`` order.
        Boolean, non-integer and out-of-range group indices are rejected; the
        original preparation and plan remain unchanged.
        """
        if isinstance(group_index, bool) or not isinstance(group_index, int):
            raise TypeError("group_index must be an integer")
        if group_index < 0 or group_index >= len(self._groups):
            raise IndexError("group_index is out of range")
        circuit = copy.copy(self._circuit)
        for qubit, pauli in zip(self._qubits, self._groups[group_index].basis):
            if pauli == "X":
                circuit.h(qubit)
            elif pauli == "Y":
                circuit.sdg(qubit)
                circuit.h(qubit)
        return circuit

    def run(
        self,
        sampler: Sampler,
        *,
        shots: int,
        parameter_bindings: Mapping[str, float] | None = None,
        seed: int | None = None,
    ) -> list[ExecutionResult]:
        """Sample the product bases in group order using one batch submission.

        Args:
            sampler: Backend implementing the public Sampler protocol.
            shots: Positive sample count for each measurement group.
            parameter_bindings: Exact finite values for all preparation symbols.
            seed: Optional non-negative seed forwarded to each group.

        Returns:
            Raw counts in ``groups`` order; identity terms require no samples.
            An empty plan returns an empty list without submitting a batch.

        Raises:
            TypeError: If a dependency, option, or sampler result has an invalid type.
            ValueError: If bindings, counts, or sampler results violate the contract.
        """
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        bound = bind_circuit_parameters(
            self._circuit, {} if parameter_bindings is None else parameter_bindings
        )
        base = SamplerRequest(bound, self._qubits, shots, seed)
        bound_plan = MeasurementPlan(
            bound, self._terms, self._groups, self._identity_indices
        )
        requests = [
            SamplerRequest(
                bound_plan.build_circuit(index),
                base.qubits,
                base.shots,
                base.seed,
            )
            for index in range(len(self._groups))
        ]
        if not requests:
            return []
        return require_sampler_results(sampler.run(requests), requests)


@final
class PauliMeasurementPlanner:
    """Create a QWC product-basis plan for an ordered Pauli sequence.

    Pauli index ``q`` maps to ``circuit.qubits[q]``, not its numeric qubit ID.
    This supports non-contiguous, reordered Core qubits. The source circuit
    must be a valid, positive-width quantum preparation without any direct
    or nested measurement, reset or dynamic operation. Pauli widths must
    match the circuit. Scalars and phases are retained but not interpreted.

    The planner copies all inputs at construction. ``none`` or ``qwc`` selects
    local grouping; global commuting groups intentionally are not accepted
    because they may require entangled joint readout rather than local basis
    rotations. Empty terms produce an empty plan.
    Parameterized preparations retain their symbols until ``plan.run`` binds
    them. All preparation qubits, including any ancillas, remain in the plan.

    Args:
        circuit: State preparation, ansatz, or existing preparation circuit.
        terms: Ordered Pauli observables with matching width.
        grouping: ``"none"`` or ``"qwc"``.
        prefix: Symbol prefix used to build a preparation or ansatz.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If the circuit, observables, widths, or grouping are invalid.
    """

    __slots__ = ("_circuit", "_grouping", "_terms")

    def __init__(
        self,
        circuit: StatePreparation | Ansatz | Circuit,
        terms: Sequence[PauliString],
        *,
        grouping: str = "qwc",
        prefix: str = "theta",
    ) -> None:
        validate_parameter_prefix(prefix)
        if isinstance(circuit, (StatePreparation, Ansatz)):
            circuit = build_component_circuit(circuit, prefix=prefix)
        elif not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a StatePreparation, Ansatz, or Circuit")
        _validate_preparation_circuit(circuit)
        self._circuit = copy.copy(circuit)
        self._terms = snapshot_terms(terms)
        if any(term.num_qubits != self._circuit.num_qubits for term in self._terms):
            raise ValueError("Pauli term width must match circuit qubit count")
        self._grouping = PauliGroupingStrategy(grouping)

    def plan(self) -> MeasurementPlan:
        """Return an independent plan with original term-index bookkeeping."""
        active_positions = tuple(
            index for index, term in enumerate(self._terms) if term.support()
        )
        identity_positions = tuple(
            index for index, term in enumerate(self._terms) if not term.support()
        )
        active_terms = tuple(self._terms[index] for index in active_positions)
        groups: list[MeasurementGroup] = []
        for local_indices in self._grouping.group(active_terms):
            originals = tuple(active_positions[index] for index in local_indices)
            basis = ["I"] * self._circuit.num_qubits
            for original in originals:
                term = self._terms[original]
                for q in range(term.num_qubits):
                    pauli = term.get_pauli(q)
                    if pauli != Pauli.i():
                        basis[q] = str(pauli)
            groups.append(MeasurementGroup(originals, tuple(basis)))
        return MeasurementPlan(
            self._circuit,
            self._terms,
            tuple(groups),
            identity_positions,
        )
