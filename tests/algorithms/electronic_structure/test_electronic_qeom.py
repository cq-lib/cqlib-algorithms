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

"""Independent algebra, solver, and integration tests for electronic qEOM."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from cqlib.qis import FermionicLadder, FermionicOp, JordanWignerMapper
from cqlib.qis.state import Statevector

from cqlib_algorithms.algorithms.electronic_structure import (
    ElectronicGroundStateResult,
    ElectronicQEOM,
    ElectronicQEOMResult,
)
from cqlib_algorithms.algorithms.electronic_structure._qeom_operators import (
    _hermitian_components,
    _product,
    _qeom_operator_matrices,
    _symmetrized_double_commutator,
)
from cqlib_algorithms.algorithms.electronic_structure._qeom_solver import (
    _solve_qeom_pencil,
)
from cqlib_algorithms.algorithms.electronic_structure.electronic_qeom import (
    _structure_residuals,
)
from cqlib_algorithms.algorithms.variational import VQEResult
from cqlib_algorithms.extension.electronic_structure import (
    ElectronicEnergy,
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    FermionicExcitation,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.state_preparation import (
    HartreeFockStatePreparation,
)
from cqlib_algorithms.protocols import (
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    OptimizerResult,
    OptimizerStatus,
)


def _ladders(num_modes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    annihilation = []
    for mode in range(num_modes):
        matrix = np.zeros((1 << num_modes, 1 << num_modes), dtype=np.complex128)
        for basis in range(1 << num_modes):
            if (basis >> mode) & 1:
                sign = -1 if (basis & ((1 << mode) - 1)).bit_count() % 2 else 1
                matrix[basis ^ (1 << mode), basis] = sign
        annihilation.append(matrix)
    return [item.conj().T for item in annihilation], annihilation


def _operator_matrix(operator: FermionicOp) -> np.ndarray:
    creation, annihilation = _ladders(operator.num_modes)
    dimension = 1 << operator.num_modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for term, coefficient in operator.terms:
        matrix = np.eye(dimension, dtype=np.complex128)
        for ladder in term:
            factors = creation if ladder.kind == "create" else annihilation
            matrix = matrix @ factors[ladder.mode]
        result += coefficient * matrix
    return result


def _symmetrized_double_matrix(
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return (
        left @ middle @ right
        + right @ middle @ left
        - 0.5 * middle @ left @ right
        - 0.5 * right @ left @ middle
        - 0.5 * left @ right @ middle
        - 0.5 * middle @ right @ left
    )


def _noninteracting_system() -> ElectronicSystem:
    occupation = ElectronicOccupation.closed_shell(2, 2)
    return ElectronicSystem(
        ElectronicIntegrals(
            0.0,
            np.diag([-1.0, 0.7]),
            np.zeros((2, 2, 2, 2)),
        ),
        occupation,
    )


def _open_shell_noninteracting_system() -> ElectronicSystem:
    return ElectronicSystem(
        ElectronicIntegrals(
            0.0,
            np.diag([-1.0, 0.7]),
            np.zeros((2, 2, 2, 2)),
        ),
        ElectronicOccupation(2, 1, 0),
    )


def _ground_result(
    system: ElectronicSystem,
    *,
    mapper: JordanWignerMapper | None,
) -> ElectronicGroundStateResult:
    circuit = HartreeFockStatePreparation(system.occupation).build_circuit("hf")
    hamiltonian = ElectronicEnergy(system.integrals).to_qubit_hamiltonian()
    energy = float(
        np.vdot(
            np.asarray(Statevector.from_circuit(circuit).data),
            hamiltonian.to_matrix()
            @ np.asarray(Statevector.from_circuit(circuit).data),
        ).real
    )
    optimizer_result = OptimizerResult(
        (),
        (),
        energy,
        0,
        OptimizerStatus.NO_PARAMETERS,
    )
    vqe_result = VQEResult(energy, {}, circuit, optimizer_result)
    auxiliary = {
        name: EstimatorResult(0.0, 0.0)
        for name in ("particle_number", "magnetization", "angular_momentum")
    }
    return ElectronicGroundStateResult(
        system,
        vqe_result,
        auxiliary,
        mapper=mapper,
    )


def test_excitation_operator_and_generator_match_independent_fock_matrices():
    for excitation in (
        FermionicExcitation(4, (0,), (2,)),
        FermionicExcitation(4, (0, 1), (2, 3)),
    ):
        creation, annihilation = _ladders(4)
        transition = np.eye(16, dtype=np.complex128)
        for target in excitation.targets:
            transition = transition @ creation[target]
        for source in reversed(excitation.sources):
            transition = transition @ annihilation[source]
        np.testing.assert_allclose(
            _operator_matrix(excitation.to_fermionic_op()),
            transition,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _operator_matrix(excitation.to_fermionic_generator()),
            1j * (transition - transition.conj().T),
            rtol=0.0,
            atol=1e-12,
        )
        assert excitation.to_fermionic_generator().equiv(
            FermionicOp.excitation_generator(
                excitation.num_spin_orbitals,
                excitation.sources,
                excitation.targets,
            )
        )


def test_fermionic_product_uses_left_then_right_operator_order():
    left = FermionicOp.creation(2, 0)
    right = FermionicOp.annihilation(2, 1)
    np.testing.assert_allclose(
        _operator_matrix(_product(left, right)),
        _operator_matrix(left) @ _operator_matrix(right),
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(
        _operator_matrix(_product(left, right)),
        _operator_matrix(_product(right, left)),
        rtol=0.0,
        atol=1e-12,
    )


def test_symmetrized_double_commutator_matches_independent_six_term_formula():
    left = FermionicOp.creation(3, 0)
    middle = FermionicOp.from_terms(
        3,
        [
            (
                (FermionicLadder.create(1), FermionicLadder.annihilate(2)),
                0.3 + 0.4j,
            ),
            (
                (FermionicLadder.create(2), FermionicLadder.annihilate(1)),
                0.3 - 0.4j,
            ),
        ],
    )
    right = FermionicOp.annihilation(3, 0)
    actual = _operator_matrix(_symmetrized_double_commutator(left, middle, right))
    expected = _symmetrized_double_matrix(
        _operator_matrix(left),
        _operator_matrix(middle),
        _operator_matrix(right),
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_nonhermitian_expectation_split_is_exact_for_complex_state():
    operator = FermionicExcitation(2, (0,), (1,)).to_fermionic_op()
    real_part, imaginary_part = _hermitian_components(operator)
    assert real_part is not None
    assert imaginary_part is not None
    state = np.array([0.0, 1.0, 1j, 0.0], dtype=np.complex128) / np.sqrt(2.0)
    expected = np.vdot(state, _operator_matrix(operator) @ state)
    reconstructed = np.vdot(state, _operator_matrix(real_part) @ state) + 1j * np.vdot(
        state,
        _operator_matrix(imaginary_part) @ state,
    )
    assert reconstructed == pytest.approx(expected, abs=1e-12)


def test_correlated_small_model_qeom_matrices_match_independent_fock_algebra():
    one_body = np.array([[-0.8, 0.17], [0.17, 0.5]])
    two_body = np.zeros((2, 2, 2, 2))
    for indices in (
        (0, 0, 1, 1),
        (1, 1, 0, 0),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
    ):
        two_body[indices] = 0.23
    hamiltonian = ElectronicEnergy(
        ElectronicIntegrals(0.11, one_body, two_body)
    ).to_fermionic_op()
    pool = (
        FermionicExcitation(4, (0,), (2,)),
        FermionicExcitation(4, (1,), (3,)),
        FermionicExcitation(4, (0, 1), (2, 3)),
    )
    actual_matrices = _qeom_operator_matrices(hamiltonian, pool)
    h_matrix = _operator_matrix(hamiltonian)
    state = np.zeros(16, dtype=np.complex128)
    state[0b0011] = np.cos(0.31)
    state[0b1100] = np.exp(0.47j) * np.sin(0.31)
    transitions = [_operator_matrix(item.to_fermionic_op()) for item in pool]

    expected = [np.zeros((3, 3), dtype=np.complex128) for _ in range(4)]
    for row, left_transition in enumerate(transitions):
        left = left_transition.conj().T
        for column, right in enumerate(transitions):
            expected[0][row, column] = np.vdot(
                state,
                _symmetrized_double_matrix(left, h_matrix, right) @ state,
            )
            expected[1][row, column] = -np.vdot(
                state,
                _symmetrized_double_matrix(
                    left,
                    h_matrix,
                    right.conj().T,
                )
                @ state,
            )
            expected[2][row, column] = np.vdot(
                state,
                (left @ right - right @ left) @ state,
            )
            right_adjoint = right.conj().T
            expected[3][row, column] = -np.vdot(
                state,
                (left @ right_adjoint - right_adjoint @ left) @ state,
            )

    for operator_matrix, expected_matrix in zip(
        actual_matrices,
        expected,
        strict=True,
    ):
        actual = np.array(
            [
                [np.vdot(state, _operator_matrix(operator) @ state) for operator in row]
                for row in operator_matrix
            ]
        )
        np.testing.assert_allclose(actual, expected_matrix, rtol=0.0, atol=1e-10)

    np.testing.assert_allclose(expected[0], expected[0].conj().T, atol=1e-10)
    np.testing.assert_allclose(expected[1], expected[1].T, atol=1e-10)
    np.testing.assert_allclose(expected[2], expected[2].conj().T, atol=1e-10)
    np.testing.assert_allclose(expected[3], -expected[3].T, atol=1e-10)


class _RecordingExactEstimator(Estimator):
    def __init__(self) -> None:
        self.batches: list[list[EstimatorRequest]] = []

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        batch = list(requests)
        self.batches.append(batch)
        exact_requests = [
            EstimatorRequest(
                request.circuit,
                request.observable,
                request.parameter_bindings,
                None,
            )
            for request in batch
        ]
        return StatevectorEstimator().run(exact_requests)


class _InvalidEstimator(Estimator):
    def __init__(self, output: object) -> None:
        self.output = output

    def run(self, requests: Sequence[EstimatorRequest], /) -> list[EstimatorResult]:
        return self.output  # type: ignore[return-value]


def test_noninteracting_qeom_gap_structure_single_batch_and_shots():
    mapper = JordanWignerMapper()
    ground = _ground_result(_noninteracting_system(), mapper=mapper)
    estimator = _RecordingExactEstimator()
    excitation = FermionicExcitation(4, (0,), (2,))
    result = ElectronicQEOM(estimator).compute(
        ground,
        excitations=(excitation,),
        num_roots=1,
        shots=97,
    )
    assert result.excitation_energies == pytest.approx([1.7], abs=1e-10)
    np.testing.assert_allclose(result.m_matrix, [[1.7]], atol=1e-10)
    np.testing.assert_allclose(result.q_matrix, [[0.0]], atol=1e-10)
    np.testing.assert_allclose(result.v_matrix, [[1.0]], atol=1e-10)
    np.testing.assert_allclose(result.w_matrix, [[0.0]], atol=1e-10)
    assert result.excitations == (excitation,)
    assert max(result.structure_residuals.values()) <= 1e-12
    assert result.metric_cutoff == pytest.approx(1e-8)
    assert result.retained_rank == 2
    assert result.positive_inertia == result.negative_inertia == 1
    assert result.discarded_metric_eigenvalues.size == 0
    assert len(estimator.batches) == 1
    assert estimator.batches[0]
    assert all(request.shots == 97 for request in estimator.batches[0])
    assert all(
        request.circuit is ground.vqe_result.optimal_circuit
        for request in estimator.batches[0]
    )
    assert result.total_energies == pytest.approx([-0.3], abs=1e-10)
    for array in (
        result.excitation_energies,
        result.amplitudes,
        result.residuals,
        result.m_matrix,
        result.q_matrix,
        result.v_matrix,
        result.w_matrix,
        result.discarded_metric_eigenvalues,
    ):
        assert array.flags.writeable is False


def test_custom_excitation_order_is_snapshotted_in_result():
    mapper = JordanWignerMapper()
    ground = _ground_result(_noninteracting_system(), mapper=mapper)
    first = FermionicExcitation(4, (1,), (3,))
    second = FermionicExcitation(4, (0,), (2,))
    supplied = [first, second]
    result = ElectronicQEOM(StatevectorEstimator()).compute(
        ground,
        excitations=supplied,
    )
    supplied.reverse()
    assert result.excitations == (first, second)


def test_default_uccsd_pool_recovers_all_noninteracting_neutral_gaps():
    mapper = JordanWignerMapper()
    ground = _ground_result(_noninteracting_system(), mapper=mapper)
    result = ElectronicQEOM(StatevectorEstimator()).compute(ground)
    np.testing.assert_allclose(
        result.excitation_energies,
        [1.7, 1.7, 3.4],
        rtol=0.0,
        atol=1e-10,
    )


def test_default_open_shell_pool_uses_alpha_occupation_and_recovers_gap():
    mapper = JordanWignerMapper()
    ground = _ground_result(_open_shell_noninteracting_system(), mapper=mapper)

    result = ElectronicQEOM(StatevectorEstimator()).compute(ground)

    assert result.excitations == (FermionicExcitation(4, (0,), (2,)),)
    np.testing.assert_allclose(
        result.excitation_energies,
        [1.7],
        rtol=0.0,
        atol=1e-10,
    )


def test_qeom_solver_handles_coupled_q_block_and_positive_negative_pairs():
    m_matrix = np.array([[2.0]], dtype=np.complex128)
    q_matrix = np.array([[0.3j]], dtype=np.complex128)
    a_matrix = np.block([[m_matrix, q_matrix], [q_matrix.conj(), m_matrix.conj()]])
    b_matrix = np.diag([1.0, -1.0]).astype(np.complex128)
    solution = _solve_qeom_pencil(
        a_matrix,
        b_matrix,
        num_roots=1,
        atol=1e-12,
        rtol=1e-10,
    )
    assert solution.energies == pytest.approx([np.sqrt(4.0 - 0.09)], abs=1e-10)
    vector = solution.amplitudes[:, 0]
    assert np.vdot(vector, b_matrix @ vector).real == pytest.approx(1.0, abs=1e-10)
    assert solution.residuals[0] <= 1e-10


def test_qeom_solver_reduces_balanced_metric_nullspace_and_rejects_coupling():
    b_matrix = np.diag([1.0, 1e-14, -1.0, -1e-14]).astype(np.complex128)
    a_matrix = np.diag([2.0, 5.0, 2.0, 5.0]).astype(np.complex128)
    solution = _solve_qeom_pencil(
        a_matrix,
        b_matrix,
        num_roots=1,
        atol=1e-12,
        rtol=1e-10,
    )
    assert solution.energies == pytest.approx([2.0], abs=1e-10)
    assert solution.metric_cutoff == pytest.approx(1e-10)
    assert solution.retained_rank == 2
    assert solution.positive_inertia == solution.negative_inertia == 1
    np.testing.assert_allclose(
        solution.discarded_metric_eigenvalues,
        [-1e-14, 1e-14],
        rtol=0.0,
        atol=1e-20,
    )
    assert solution.discarded_metric_eigenvalues.flags.writeable is False

    coupled = a_matrix.copy()
    coupled[0, 1] = coupled[1, 0] = 0.1
    with pytest.raises(ValueError, match="nullspace is coupled"):
        _solve_qeom_pencil(
            coupled,
            b_matrix,
            num_roots=1,
            atol=1e-12,
            rtol=1e-10,
        )


def test_qeom_rejects_legacy_mapper_ambiguity_and_invalid_inputs():
    system = _noninteracting_system()
    legacy = _ground_result(system, mapper=None)
    excitation = FermionicExcitation(4, (0,), (2,))
    with pytest.raises(ValueError, match="require an explicit mapper"):
        ElectronicQEOM(StatevectorEstimator()).compute(
            legacy,
            excitations=(excitation,),
        )
    result = ElectronicQEOM(
        StatevectorEstimator(), mapper=JordanWignerMapper()
    ).compute(legacy, excitations=(excitation,))
    assert result.excitation_energies == pytest.approx([1.7], abs=1e-10)

    with pytest.raises(TypeError, match="ground_state"):
        ElectronicQEOM(StatevectorEstimator()).compute(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        ElectronicQEOM(StatevectorEstimator(), mapper=JordanWignerMapper()).compute(
            legacy,
            excitations=(),
        )
    with pytest.raises(ValueError, match="mode count"):
        ElectronicQEOM(StatevectorEstimator(), mapper=JordanWignerMapper()).compute(
            legacy,
            excitations=(FermionicExcitation(2, (0,), (1,)),),
        )
    with pytest.raises(ValueError, match="duplicates"):
        ElectronicQEOM(StatevectorEstimator(), mapper=JordanWignerMapper()).compute(
            legacy,
            excitations=(excitation, excitation),
        )
    with pytest.raises(TypeError, match="num_roots"):
        ElectronicQEOM(StatevectorEstimator(), mapper=JordanWignerMapper()).compute(
            legacy,
            excitations=(excitation,),
            num_roots=True,
        )
    with pytest.raises(ValueError, match="positive"):
        ElectronicQEOM(StatevectorEstimator(), mapper=JordanWignerMapper()).compute(
            legacy,
            excitations=(excitation,),
            num_roots=0,
        )
    with pytest.raises(ValueError, match="exceeds"):
        ElectronicQEOM(StatevectorEstimator(), mapper=JordanWignerMapper()).compute(
            legacy,
            excitations=(excitation,),
            num_roots=2,
        )

    with pytest.raises(TypeError, match="estimator"):
        ElectronicQEOM(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="atol"):
        ElectronicQEOM(StatevectorEstimator(), atol=0.0)


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        ((), TypeError, "return a list"),
        ([], ValueError, "one result"),
        ([object(), object()], TypeError, "EstimatorResult"),
    ],
)
def test_qeom_rejects_invalid_estimator_responses(output, error, message):
    ground = _ground_result(_noninteracting_system(), mapper=JordanWignerMapper())
    with pytest.raises(error, match=message):
        ElectronicQEOM(_InvalidEstimator(output)).compute(
            ground,
            excitations=(FermionicExcitation(4, (0,), (2,)),),
        )


def test_result_defensively_copies_arrays_and_validates_shapes():
    ground = _ground_result(_noninteracting_system(), mapper=JordanWignerMapper())
    excitation = FermionicExcitation(4, (0,), (2,))
    energies = np.array([1.0])
    amplitudes = np.array([[1.0], [0.0]], dtype=np.complex128)
    residuals = np.array([0.0])
    matrix = np.array([[1.0]], dtype=np.complex128)
    structure_residuals = {"M": 0.0, "Q": 0.0, "V": 0.0, "W": 0.0}
    discarded = np.empty(0, dtype=np.float64)
    result = ElectronicQEOMResult(
        ground,
        (excitation,),
        energies,
        amplitudes,
        residuals,
        structure_residuals,
        1e-10,
        2,
        1,
        1,
        discarded,
        matrix,
        np.zeros((1, 1), dtype=np.complex128),
        matrix,
        np.zeros((1, 1), dtype=np.complex128),
    )
    energies[0] = 9.0
    amplitudes[0, 0] = 9.0
    structure_residuals["M"] = 9.0
    assert result.excitation_energies[0] == 1.0
    assert result.amplitudes[0, 0] == 1.0
    assert result.structure_residuals["M"] == 0.0
    with pytest.raises(TypeError):
        result.structure_residuals["M"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="shape"):
        ElectronicQEOMResult(
            ground,
            (excitation,),
            np.array([1.0]),
            np.ones((2, 1), dtype=np.complex128),
            np.array([0.0]),
            {"M": 0.0, "Q": 0.0, "V": 0.0, "W": 0.0},
            1e-10,
            2,
            1,
            1,
            np.empty(0),
            np.eye(2, dtype=np.complex128),
            matrix,
            matrix,
            matrix,
        )

    with pytest.raises(ValueError, match="must not be empty"):
        ElectronicQEOMResult(
            ground,
            (),
            np.array([1.0]),
            np.empty((0, 1), dtype=np.complex128),
            np.array([0.0]),
            {"M": 0.0, "Q": 0.0, "V": 0.0, "W": 0.0},
            1e-10,
            0,
            0,
            0,
            np.empty(0),
            np.empty((0, 0), dtype=np.complex128),
            np.empty((0, 0), dtype=np.complex128),
            np.empty((0, 0), dtype=np.complex128),
            np.empty((0, 0), dtype=np.complex128),
        )

    with pytest.raises(ValueError, match="amplitude rows"):
        ElectronicQEOMResult(
            ground,
            (excitation,),
            np.array([1.0]),
            np.empty((0, 1), dtype=np.complex128),
            np.array([0.0]),
            {"M": 0.0, "Q": 0.0, "V": 0.0, "W": 0.0},
            1e-10,
            2,
            1,
            1,
            np.empty(0),
            matrix,
            np.zeros((1, 1), dtype=np.complex128),
            matrix,
            np.zeros((1, 1), dtype=np.complex128),
        )


def test_structure_residuals_expose_preprojection_inconsistency():
    m_matrix = np.array([[0.0, 4.0], [0.0, 0.0]], dtype=np.complex128)
    q_matrix = np.array([[0.0, 3.0], [0.0, 0.0]], dtype=np.complex128)
    v_matrix = np.array([[0.0, 2.0j], [0.0, 0.0]], dtype=np.complex128)
    w_matrix = np.eye(2, dtype=np.complex128)
    diagnostics = _structure_residuals(
        m_matrix,
        q_matrix,
        v_matrix,
        w_matrix,
    )
    assert diagnostics == pytest.approx({"M": 2.0, "Q": 1.5, "V": 1.0, "W": 1.0})


def test_result_defensively_copies_discarded_metric_spectrum():
    ground = _ground_result(_noninteracting_system(), mapper=JordanWignerMapper())
    excitations = (
        FermionicExcitation(4, (0,), (2,)),
        FermionicExcitation(4, (1,), (3,)),
    )
    discarded = np.array([-1e-14, 1e-14])
    result = ElectronicQEOMResult(
        ground,
        excitations,
        np.array([1.0]),
        np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.complex128),
        np.array([0.0]),
        {"M": 0.0, "Q": 0.0, "V": 0.0, "W": 0.0},
        1e-10,
        2,
        1,
        1,
        discarded,
        np.eye(2, dtype=np.complex128),
        np.zeros((2, 2), dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
        np.zeros((2, 2), dtype=np.complex128),
    )
    discarded[:] = 0.0
    np.testing.assert_allclose(
        result.discarded_metric_eigenvalues,
        [-1e-14, 1e-14],
        rtol=0.0,
        atol=0.0,
    )
