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

"""Neutral particle-number-conserving quantum equation-of-motion workflow."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import final

import numpy as np
from cqlib.qis import FermionToQubitMapper

from ..._utils.hamiltonian import snapshot_real_hamiltonian
from ...extension.electronic_structure.electronic import ElectronicEnergy
from ...extension.electronic_structure.fermionic import (
    FermionicExcitation,
    generate_uccsd_excitations,
)
from ...protocols.estimator import (
    Estimator,
    EstimatorRequest,
    require_estimator_results,
)
from ._qeom_operators import _hermitian_components, _qeom_operator_matrices
from ._qeom_solver import _solve_qeom_pencil
from .results import ElectronicGroundStateResult, ElectronicQEOMResult


def _normalize_tolerance(value: float, *, name: str) -> float:
    """Validate and return a positive finite tolerance."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


def _snapshot_pool(
    excitations: Sequence[FermionicExcitation] | None,
    ground_state: ElectronicGroundStateResult,
) -> tuple[FermionicExcitation, ...]:
    if excitations is None:
        pool = generate_uccsd_excitations(ground_state.system.occupation)
    else:
        if isinstance(excitations, (str, bytes)) or not isinstance(
            excitations, Sequence
        ):
            raise TypeError("excitations must be a sequence or None")
        pool = tuple(excitations)
    if not pool:
        raise ValueError("the qEOM excitation pool must not be empty")
    for excitation in pool:
        if not isinstance(excitation, FermionicExcitation):
            raise TypeError(
                "the qEOM excitation pool must contain FermionicExcitation values"
            )
        if excitation.num_spin_orbitals != ground_state.system.num_spin_orbitals:
            raise ValueError(
                "qEOM excitations must match the electronic-system mode count"
            )
    if len(set(pool)) != len(pool):
        raise ValueError("the qEOM excitation pool must not contain duplicates")
    return pool


def _structure_residuals(
    m_matrix: np.ndarray,
    q_matrix: np.ndarray,
    v_matrix: np.ndarray,
    w_matrix: np.ndarray,
) -> dict[str, float]:
    """Return pre-projection spectral-norm residuals for qEOM block symmetries."""
    return {
        "M": float(np.linalg.norm(m_matrix - m_matrix.conj().T, ord=2)) / 2.0,
        "Q": float(np.linalg.norm(q_matrix - q_matrix.T, ord=2)) / 2.0,
        "V": float(np.linalg.norm(v_matrix - v_matrix.conj().T, ord=2)) / 2.0,
        "W": float(np.linalg.norm(w_matrix + w_matrix.T, ord=2)) / 2.0,
    }


@final
class ElectronicQEOM:
    """Compute neutral excitation energies from an existing ground-state result.

    Matrix elements use the symmetrized double commutator
    ``[A,B,C]=([[A,B],C]+[A,[B,C]])/2``. Every non-Hermitian expectation is
    split exactly into Hermitian real and imaginary components, and all
    nonzero components are submitted in one estimator batch.

    Args:
        estimator: Expectation-value estimator for qEOM matrix elements.
        mapper: Optional mapper override. It must match mapper provenance in
            the ground-state result when both are present.
        atol: Positive absolute tolerance for the generalized eigensolver.
        rtol: Positive relative tolerance for the generalized eigensolver.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a tolerance is not finite and positive.
    """

    __slots__ = ("_atol", "_estimator", "_mapper", "_rtol")

    def __init__(
        self,
        estimator: Estimator,
        *,
        mapper: FermionToQubitMapper | None = None,
        atol: float = 1e-10,
        rtol: float = 1e-8,
    ) -> None:
        if not isinstance(estimator, Estimator):
            raise TypeError("estimator must be an Estimator")
        if mapper is not None and not isinstance(mapper, FermionToQubitMapper):
            raise TypeError("mapper must implement FermionToQubitMapper or be None")
        self._estimator = estimator
        self._mapper = mapper
        self._atol = _normalize_tolerance(atol, name="atol")
        self._rtol = _normalize_tolerance(rtol, name="rtol")

    def _resolve_mapper(
        self,
        ground_state: ElectronicGroundStateResult,
    ) -> FermionToQubitMapper:
        provenance = ground_state.mapper
        if provenance is None:
            if self._mapper is None:
                raise ValueError(
                    "legacy ground-state results without mapper provenance require "
                    "an explicit mapper"
                )
            return self._mapper
        if self._mapper is not None and self._mapper is not provenance:
            raise ValueError(
                "explicit mapper must be the mapper recorded by the ground state"
            )
        return provenance

    def run(
        self,
        ground_state: ElectronicGroundStateResult,
        *,
        excitations: Sequence[FermionicExcitation] | None = None,
        num_roots: int | None = None,
        shots: int | None = None,
    ) -> ElectronicQEOMResult:
        """Measure qEOM matrices and solve for validated positive roots.

        Args:
            ground_state: Previously computed electronic ground-state result.
            excitations: Optional neutral excitation pool. The UCCSD pool is
                generated from the stored occupation when omitted.
            num_roots: Optional positive limit on returned excitation roots.
            shots: Optional finite-shot count forwarded to the estimator.

        Returns:
            Excitation energies, amplitudes, matrices, and residual diagnostics.

        Raises:
            TypeError: If an input or estimator result has the wrong type.
            ValueError: If mapper provenance, dimensions, pool contents, or
                estimator result counts are inconsistent.
        """
        if not isinstance(ground_state, ElectronicGroundStateResult):
            raise TypeError("ground_state must be an ElectronicGroundStateResult")
        if num_roots is not None:
            if isinstance(num_roots, bool) or not isinstance(num_roots, int):
                raise TypeError("num_roots must be an integer or None")
            if num_roots < 1:
                raise ValueError("num_roots must be positive")
        mapper = self._resolve_mapper(ground_state)
        pool = _snapshot_pool(excitations, ground_state)
        fermionic_hamiltonian = ElectronicEnergy(
            ground_state.system.integrals
        ).to_fermionic_op()
        operator_matrices = _qeom_operator_matrices(
            fermionic_hamiltonian,
            pool,
        )

        expected_width = mapper.num_qubits(fermionic_hamiltonian.num_modes)
        if isinstance(expected_width, bool) or not isinstance(expected_width, int):
            raise TypeError("mapper num_qubits must return an int")
        if expected_width < 1:
            raise ValueError("mapper num_qubits must return a positive width")
        circuit = ground_state.vqe_result.optimal_circuit
        circuit.validate()
        if circuit.used_symbols:
            raise ValueError("ground-state circuit must be fully bound")
        if circuit.num_qubits != expected_width:
            raise ValueError(
                "ground-state circuit width must match the recorded mapper"
            )

        pool_size = len(pool)
        values = [
            np.zeros((pool_size, pool_size), dtype=np.complex128) for _ in range(4)
        ]
        requests: list[EstimatorRequest] = []
        destinations: list[tuple[int, int, int, bool]] = []
        for matrix_index, matrix in enumerate(operator_matrices):
            for row, operator_row in enumerate(matrix):
                for column, operator in enumerate(operator_row):
                    real_part, imaginary_part = _hermitian_components(operator)
                    for is_imaginary, component in (
                        (False, real_part),
                        (True, imaginary_part),
                    ):
                        if component is None:
                            continue
                        mapped = mapper.map_operator(component)
                        observable = snapshot_real_hamiltonian(
                            mapped,
                            expected_width=expected_width,
                            name="qEOM matrix component",
                        )
                        requests.append(
                            EstimatorRequest(circuit, observable, {}, shots)
                        )
                        destinations.append((matrix_index, row, column, is_imaginary))

        estimates = require_estimator_results(self._estimator.run(requests), requests)
        for destination, estimate in zip(destinations, estimates, strict=True):
            matrix_index, row, column, is_imaginary = destination
            contribution = (
                1j * estimate.expectation if is_imaginary else estimate.expectation
            )
            values[matrix_index][row, column] += contribution

        m_matrix, q_matrix, v_matrix, w_matrix = values
        structure_residuals = _structure_residuals(
            m_matrix,
            q_matrix,
            v_matrix,
            w_matrix,
        )
        m_matrix = 0.5 * (m_matrix + m_matrix.conj().T)
        q_matrix = 0.5 * (q_matrix + q_matrix.T)
        v_matrix = 0.5 * (v_matrix + v_matrix.conj().T)
        w_matrix = 0.5 * (w_matrix - w_matrix.T)
        a_matrix = np.block([[m_matrix, q_matrix], [q_matrix.conj(), m_matrix.conj()]])
        b_matrix = np.block(
            [[v_matrix, w_matrix], [-w_matrix.conj(), -v_matrix.conj()]]
        )
        solution = _solve_qeom_pencil(
            a_matrix,
            b_matrix,
            num_roots=num_roots,
            atol=self._atol,
            rtol=self._rtol,
        )
        return ElectronicQEOMResult(
            ground_state,
            pool,
            solution.energies,
            solution.amplitudes,
            solution.residuals,
            structure_residuals,
            solution.metric_cutoff,
            solution.retained_rank,
            solution.positive_inertia,
            solution.negative_inertia,
            solution.discarded_metric_eigenvalues,
            m_matrix,
            q_matrix,
            v_matrix,
            w_matrix,
        )

    def compute(
        self,
        ground_state: ElectronicGroundStateResult,
        *,
        excitations: Sequence[FermionicExcitation] | None = None,
        num_roots: int | None = None,
        shots: int | None = None,
    ) -> ElectronicQEOMResult:
        """Solve qEOM with the same inputs and result as :meth:`run`."""
        return self.run(
            ground_state, excitations=excitations, num_roots=num_roots, shots=shots
        )


__all__ = ["ElectronicQEOM"]
