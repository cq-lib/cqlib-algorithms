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

"""Optional end-to-end molecular reference checks."""

from collections.abc import Sequence

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci, gto, mcscf, mp, scf

from cqlib_algorithms.algorithms import VQE, ElectronicGroundState, ElectronicQEOM
from cqlib_algorithms.extension.electronic_structure import (
    ActiveSpace,
    AngularMomentum,
    Atom,
    ElectronicEnergy,
    ElectronicOccupation,
    ElectronicSystem,
    MolecularSpecification,
    PySCFDriver,
    SpinResolvedActiveSpace,
    SpinResolvedElectronicStructureData,
    hartree_fock_initial_point,
    mp2_initial_point,
    transform_active_space,
)
from cqlib_algorithms.operators.ansatz import (
    PUCCDAnsatz,
    PUCCSDAnsatz,
    UCCSDAnsatz,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.state_preparation import (
    HartreeFockStatePreparation,
)
from cqlib_algorithms.protocols import (
    Estimator,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)


class _FixedPointOptimizer(Optimizer):
    """Evaluate each supplied point once without changing it."""

    def run(
        self,
        estimator: Estimator,
        requests: Sequence[OptimizerRequest],
        /,
    ) -> list[OptimizerResult]:
        batch = list(requests)
        estimates = estimator.run([request.estimator_request for request in batch])
        return [
            OptimizerResult(
                request.parameter_names,
                tuple(
                    request.estimator_request.parameter_bindings[name]
                    for name in request.parameter_names
                ),
                estimate.expectation,
                0,
                OptimizerStatus.STEP_TOLERANCE,
            )
            for request, estimate in zip(batch, estimates, strict=True)
        ]


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_helium_hydride_uhf_hamiltonian_and_spin_match_pyscf():
    distance = 1.4
    specification = MolecularSpecification(
        atoms=(Atom("He", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, distance))),
        basis="sto-3g",
        multiplicity=2,
    )
    data = PySCFDriver(reference="uhf").run(specification)
    assert isinstance(data, SpinResolvedElectronicStructureData)
    system = ElectronicSystem.from_data(data)
    hamiltonian = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    spin_squared = (
        AngularMomentum(
            system.num_spatial_orbitals,
            system.alpha_beta_overlap,
        )
        .to_qubit_hamiltonian()
        .to_matrix()
    )
    reference_state = (
        HartreeFockStatePreparation(system.occupation)
        .build_circuit("uhf")
        .to_matrix()[:, 0]
    )

    molecule = gto.M(
        atom=f"He 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        spin=1,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.UHF(molecule).run()
    reference_spin_squared = mean_field.spin_square()[0]

    assert np.vdot(
        reference_state, hamiltonian @ reference_state
    ).real == pytest.approx(
        mean_field.e_tot,
        abs=1e-9,
    )
    assert np.vdot(
        reference_state, spin_squared @ reference_state
    ).real == pytest.approx(
        reference_spin_squared,
        abs=1e-9,
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_lithium_uhf_active_space_ground_state_preserves_reference_energy():
    specification = MolecularSpecification(
        atoms=(Atom("Li", (0.0, 0.0, 0.0)),),
        basis="sto-3g",
        multiplicity=2,
    )
    data = PySCFDriver(reference="uhf").run(specification)
    assert isinstance(data, SpinResolvedElectronicStructureData)
    selection = SpinResolvedActiveSpace.paired(
        frozen_orbitals=(0,),
        active_orbitals=(1, 2),
    )
    system = ElectronicSystem.from_data(data, selection)

    result = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="uhf_active",
    ).compute(
        system,
        hartree_fock_initial_point(system, prefix="uhf_active"),
    )

    molecule = gto.M(
        atom="Li 0 0 0",
        basis="sto-3g",
        spin=1,
        unit="Angstrom",
        verbose=0,
    )
    reference = scf.UHF(molecule).run()
    assert reference.converged
    assert result.total_energy == pytest.approx(reference.e_tot, abs=1e-9)
    assert result.total_energy == pytest.approx(data.hartree_fock_energy, abs=1e-9)
    assert result.particle_number.expectation == pytest.approx(1.0, abs=1e-10)
    assert result.magnetization.expectation == pytest.approx(0.5, abs=1e-10)
    assert result.hamiltonian_constant == pytest.approx(
        result.nuclear_repulsion_energy + result.inactive_energy,
        abs=1e-12,
    )
    assert result.total_energy == pytest.approx(
        result.hamiltonian_constant + result.active_space_energy,
        abs=1e-12,
    )


def _fixed_particle_ground_energy(matrix: np.ndarray, num_electrons: int) -> float:
    indices = [
        index for index in range(matrix.shape[0]) if index.bit_count() == num_electrons
    ]
    block = matrix[np.ix_(indices, indices)]
    return float(np.linalg.eigvalsh(block)[0])


def _fixed_spin_sector(
    matrix: np.ndarray,
    num_spatial_orbitals: int,
    num_alpha: int,
    num_beta: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    alpha_modes = tuple(range(0, 2 * num_spatial_orbitals, 2))
    beta_modes = tuple(range(1, 2 * num_spatial_orbitals, 2))
    indices = [
        basis
        for basis in range(matrix.shape[0])
        if sum(bool(basis & (1 << mode)) for mode in alpha_modes) == num_alpha
        and sum(bool(basis & (1 << mode)) for mode in beta_modes) == num_beta
    ]
    values, vectors = np.linalg.eigh(matrix[np.ix_(indices, indices)])
    return indices, values, vectors


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _double_transition_amplitude(
    source_index: int,
    target_index: int,
) -> float:
    """Return <target|a2+ a3+ a1 a0|source> independently."""
    state = source_index
    amplitude = 1.0
    for create, mode in ((False, 0), (False, 1), (True, 3), (True, 2)):
        applied = _apply_ladder(state, mode, create=create)
        assert applied is not None
        state, sign = applied
        amplitude *= sign
    assert state == target_index
    return amplitude


def _operator_matrix(
    num_modes: int,
    operations: tuple[tuple[bool, int], ...],
) -> np.ndarray:
    size = 1 << num_modes
    result = np.zeros((size, size), dtype=np.complex128)
    for basis in range(size):
        output = basis
        amplitude = 1.0
        for create, mode in reversed(operations):
            applied = _apply_ladder(output, mode, create=create)
            if applied is None:
                break
            output, sign = applied
            amplitude *= sign
        else:
            result[output, basis] = amplitude
    return result


def _independent_two_orbital_singlet_model(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = _operator_matrix(4, ((True, 2), (False, 0)))
    beta = _operator_matrix(4, ((True, 3), (False, 1)))
    pair = _operator_matrix(
        4,
        ((True, 2), (True, 3), (False, 1), (False, 0)),
    )
    single_generator = alpha - alpha.conj().T + beta - beta.conj().T
    pair_generator = pair - pair.conj().T

    reference = np.eye(16, dtype=np.complex128)[:, 0b0011]
    open_shell = single_generator @ reference
    open_shell /= np.linalg.norm(open_shell)
    paired = pair_generator @ reference
    paired /= np.linalg.norm(paired)
    basis = np.column_stack((reference, open_shell, paired))
    return (
        basis,
        basis.conj().T @ matrix @ basis,
        basis.conj().T @ single_generator @ basis,
        basis.conj().T @ pair_generator @ basis,
    )


def _matrix_exponential(generator: np.ndarray, theta: float) -> np.ndarray:
    values, vectors = np.linalg.eig(generator)
    return vectors @ np.diag(np.exp(theta * values)) @ np.linalg.inv(vectors)


def _paired_reference_angles(
    matrix: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    basis, singlet_hamiltonian, single_generator, pair_generator = (
        _independent_two_orbital_singlet_model(matrix)
    )
    pair_block = singlet_hamiltonian[np.ix_((0, 2), (0, 2))]
    _, pair_vectors = np.linalg.eigh(pair_block)
    pair_ground = pair_vectors[:, 0]
    if pair_ground[0].real < 0.0:
        pair_ground = -pair_ground
    pair_theta = float(np.arctan2(pair_ground[1].real, pair_ground[0].real))

    _, singlet_vectors = np.linalg.eigh(singlet_hamiltonian)
    singlet_ground = singlet_vectors[:, 0]
    if singlet_ground[0].real < 0.0:
        singlet_ground = -singlet_ground
    scaled_open = np.sqrt(2.0) * float(singlet_ground[1].real)
    assert abs(scaled_open) <= 1.0 + 1e-12
    single_theta = 0.5 * float(np.arcsin(np.clip(scaled_open, -1.0, 1.0)))
    after_single = _matrix_exponential(single_generator, single_theta)[:, 0]
    pair_rotation = float(
        np.arctan2(singlet_ground[2].real, singlet_ground[0].real)
        - np.arctan2(after_single[2].real, after_single[0].real)
    )
    expected = basis @ singlet_ground
    actual = basis @ (_matrix_exponential(pair_generator, pair_rotation) @ after_single)
    assert abs(np.vdot(expected, actual)) == pytest.approx(1.0, abs=1e-12)
    return pair_theta, single_theta, pair_rotation, expected


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("distance", [0.5, 0.7414, 1.5])
def test_h2_hf_and_fci_energies(distance):
    specification = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, distance))),
        basis="sto-3g",
    )
    result = PySCFDriver().run(specification)
    matrix = ElectronicEnergy(result.integrals).to_qubit_hamiltonian().to_matrix()
    occupation = ElectronicOccupation(
        result.num_spatial_orbitals,
        result.num_alpha,
        result.num_beta,
    )
    hf_state = (
        HartreeFockStatePreparation(occupation).build_circuit("hf").to_matrix()[:, 0]
    )

    occupied_index = 0b0011
    assert matrix[occupied_index, occupied_index].real == pytest.approx(
        result.hartree_fock_energy, abs=1e-10
    )
    assert np.vdot(hf_state, matrix @ hf_state).real == pytest.approx(
        result.hartree_fock_energy, abs=1e-10
    )
    reference_fci = fci.direct_spin1.kernel(
        result.integrals.one_body,
        result.integrals.two_body,
        result.num_spatial_orbitals,
        (result.num_alpha, result.num_beta),
        ecore=result.integrals.constant,
    )[0]
    assert _fixed_particle_ground_energy(matrix, result.num_electrons) == pytest.approx(
        reference_fci, abs=1e-9
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_h2_uccsd_exact_angle_matches_fci_and_vqe_fixed_point():
    specification = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.7414))),
        basis="sto-3g",
    )
    result = PySCFDriver().run(specification)
    hamiltonian = ElectronicEnergy(result.integrals).to_qubit_hamiltonian()
    matrix = hamiltonian.to_matrix()
    occupation = ElectronicOccupation.closed_shell(
        result.num_spatial_orbitals,
        result.num_electrons,
    )
    ansatz = UCCSDAnsatz(occupation)

    hf_index = 0b0011
    double_index = 0b1100
    block = matrix[np.ix_([hf_index, double_index], [hf_index, double_index])]
    values, vectors = np.linalg.eigh(block)
    ground = vectors[:, 0]
    if ground[0].real < 0.0:
        ground = -ground
    transition_sign = _double_transition_amplitude(hf_index, double_index)
    theta = float(
        np.arctan2(
            ground[1].real / transition_sign,
            ground[0].real,
        )
    )

    circuit = ansatz.build_circuit("chem")
    bindings = {
        "chem_theta_0": 0.0,
        "chem_theta_1": 0.0,
        "chem_theta_2": theta,
    }
    state = circuit.assign_parameters(bindings).to_matrix()[:, 0]
    direct_energy = float(np.vdot(state, matrix @ state).real)
    reference_fci = fci.direct_spin1.kernel(
        result.integrals.one_body,
        result.integrals.two_body,
        result.num_spatial_orbitals,
        (1, 1),
        ecore=result.integrals.constant,
    )[0]

    assert values[0] == pytest.approx(reference_fci, abs=1e-9)
    assert direct_energy == pytest.approx(reference_fci, abs=1e-9)

    vqe_result = VQE(
        StatevectorEstimator(),
        ansatz,
        _FixedPointOptimizer(),
        prefix="chem",
    ).compute_minimum_eigenvalue(hamiltonian, bindings)
    assert vqe_result.eigenvalue == pytest.approx(reference_fci, abs=1e-9)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_h2_mp2_initial_point_and_ground_state_workflow_match_references():
    distance = 0.7414
    specification = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, distance))),
        basis="sto-3g",
    )
    data = PySCFDriver().run(specification)
    system = ElectronicSystem.from_data(data)
    initial_point = mp2_initial_point(system, prefix="chem")

    molecule = gto.M(
        atom=(("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distance))),
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run()
    mp2_solver = mp.MP2(mean_field).run()
    assert initial_point["chem_theta_2"] == pytest.approx(
        float(mp2_solver.t2[0, 0, 0, 0]), abs=1e-10
    )

    matrix = ElectronicEnergy(data.integrals).to_qubit_hamiltonian().to_matrix()
    point, reference_fci = (
        _paired_reference_angles(matrix)[0],
        fci.direct_spin1.kernel(
            data.integrals.one_body,
            data.integrals.two_body,
            2,
            (1, 1),
            ecore=data.integrals.constant,
        )[0],
    )
    exact_initial = {
        "chem_theta_0": 0.0,
        "chem_theta_1": 0.0,
        "chem_theta_2": point,
    }
    result = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="chem",
    ).compute(system, exact_initial)
    assert result.total_energy == pytest.approx(reference_fci, abs=1e-9)
    assert result.particle_number.expectation == pytest.approx(2.0, abs=1e-10)
    assert result.magnetization.expectation == pytest.approx(0.0, abs=1e-10)
    assert result.angular_momentum.expectation == pytest.approx(0.0, abs=1e-10)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_h2_cation_rohf_uccsd_ground_state_and_qeom_chain():
    distance = 1.1
    specification = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, distance))),
        basis="sto-3g",
        charge=1,
        multiplicity=2,
    )
    data = PySCFDriver().run(specification)
    system = ElectronicSystem.from_data(data)
    hamiltonian = ElectronicEnergy(data.integrals).to_qubit_hamiltonian()
    matrix = hamiltonian.to_matrix()
    indices, exact_values, _ = _fixed_spin_sector(matrix, 2, 1, 0)
    reference_fci = fci.direct_spin1.kernel(
        data.integrals.one_body,
        data.integrals.two_body,
        data.num_spatial_orbitals,
        (1, 0),
        ecore=data.integrals.constant,
    )[0]

    assert (data.num_alpha, data.num_beta) == (1, 0)
    assert indices == [0b0001, 0b0100]
    assert exact_values[0] == pytest.approx(reference_fci, abs=1e-9)
    occupation = ElectronicOccupation(2, 1, 0)
    hf_state = (
        HartreeFockStatePreparation(occupation).build_circuit("hf").to_matrix()[:, 0]
    )
    assert float(np.vdot(hf_state, matrix @ hf_state).real) == pytest.approx(
        data.hartree_fock_energy,
        abs=1e-9,
    )

    ansatz = UCCSDAnsatz(system.occupation)
    circuit = ansatz.build_circuit("open")
    assert circuit.used_symbols == ["open_theta_0"]
    ground = ElectronicGroundState(
        StatevectorEstimator(),
        _FixedPointOptimizer(),
        prefix="open",
    ).compute(system, {"open_theta_0": 0.0})
    assert ground.total_energy == pytest.approx(reference_fci, abs=1e-9)
    assert ground.particle_number.expectation == pytest.approx(1.0, abs=1e-10)
    assert ground.magnetization.expectation == pytest.approx(0.5, abs=1e-10)
    assert ground.angular_momentum.expectation == pytest.approx(0.75, abs=1e-10)

    excited = ElectronicQEOM(StatevectorEstimator()).compute(ground, num_roots=1)
    assert excited.excitation_energies == pytest.approx(
        [exact_values[1] - exact_values[0]],
        abs=1e-8,
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_helium_hydride_rohf_frozen_pair_matches_open_shell_casci():
    distance = 1.4
    specification = MolecularSpecification(
        atoms=(Atom("He", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, distance))),
        basis="sto-3g",
        multiplicity=2,
    )
    data = PySCFDriver().run(specification)
    active_space = ActiveSpace.manual(
        frozen_orbitals=(0,),
        active_orbitals=tuple(range(1, data.num_spatial_orbitals)),
    )
    system = ElectronicSystem.from_data(data, active_space)
    matrix = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    hf_state = (
        HartreeFockStatePreparation(system.occupation)
        .build_circuit("active_hf")
        .to_matrix()[:, 0]
    )

    molecule = gto.M(
        atom=f"He 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        spin=1,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.ROHF(molecule).run()
    casci_energy = mcscf.CASCI(
        mean_field,
        data.num_spatial_orbitals - 1,
        (1, 0),
    ).kernel()[0]

    assert (data.num_alpha, data.num_beta) == (2, 1)
    assert system.occupation == ElectronicOccupation(
        data.num_spatial_orbitals - 1,
        1,
        0,
    )
    assert float(np.vdot(hf_state, matrix @ hf_state).real) == pytest.approx(
        casci_energy,
        abs=1e-8,
    )
    assert casci_energy == pytest.approx(data.hartree_fock_energy, abs=1e-8)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_h2_paired_ansatze_reach_independent_fci_reference():
    specification = MolecularSpecification(
        atoms=(Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.7414))),
        basis="sto-3g",
    )
    result = PySCFDriver().run(specification)
    matrix = ElectronicEnergy(result.integrals).to_qubit_hamiltonian().to_matrix()
    occupation = ElectronicOccupation.closed_shell(2, 2)
    pair_theta, single_theta, pair_rotation, expected_singlet = (
        _paired_reference_angles(matrix)
    )
    reference_fci = fci.direct_spin1.kernel(
        result.integrals.one_body,
        result.integrals.two_body,
        2,
        (1, 1),
        ecore=result.integrals.constant,
    )[0]

    puccd_state = (
        PUCCDAnsatz(occupation)
        .build_circuit("pair")
        .assign_parameters({"pair_theta_0": pair_theta})
        .to_matrix()[:, 0]
    )
    puccsd_state = (
        PUCCSDAnsatz(occupation)
        .build_circuit("spin")
        .assign_parameters(
            {
                "spin_theta_0": single_theta,
                "spin_theta_1": pair_rotation,
            }
        )
        .to_matrix()[:, 0]
    )
    assert float(np.vdot(puccd_state, matrix @ puccd_state).real) == pytest.approx(
        reference_fci, abs=1e-9
    )
    assert float(np.vdot(puccsd_state, matrix @ puccsd_state).real) == pytest.approx(
        reference_fci, abs=1e-9
    )
    assert abs(np.vdot(expected_singlet, puccsd_state)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.slow
def test_lih_frozen_core_matches_casci():
    specification = MolecularSpecification(
        atoms=(Atom("Li", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 1.6))),
        basis="sto-3g",
    )
    result = PySCFDriver().run(specification)
    active = transform_active_space(
        result.integrals,
        ElectronicOccupation(
            result.num_spatial_orbitals,
            result.num_alpha,
            result.num_beta,
        ),
        ActiveSpace.manual(frozen_orbitals=(0,), active_orbitals=(1, 2)),
    )
    matrix = ElectronicEnergy(active.integrals).to_qubit_hamiltonian().to_matrix()
    active_occupation = active.occupation
    active_hf_state = (
        HartreeFockStatePreparation(active_occupation)
        .build_circuit("active_hf")
        .to_matrix()[:, 0]
    )
    active_uccsd = UCCSDAnsatz(active_occupation)
    active_circuit = active_uccsd.build_circuit("active")
    zero_bindings = {name: 0.0 for name in active_circuit.used_symbols}
    uccsd_zero_state = active_circuit.assign_parameters(zero_bindings).to_matrix()[:, 0]
    random_bindings = dict(
        zip(active_circuit.used_symbols, (0.19, -0.31, 0.47), strict=True)
    )
    uccsd_random_state = active_circuit.assign_parameters(random_bindings).to_matrix()[
        :, 0
    ]

    molecule = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run()
    casci_energy = mcscf.CASCI(mean_field, 2, 2).kernel()[0]
    pair_theta, single_theta, pair_rotation, expected_singlet = (
        _paired_reference_angles(matrix)
    )
    puccd_state = (
        PUCCDAnsatz(active_occupation)
        .build_circuit("pair")
        .assign_parameters({"pair_theta_0": pair_theta})
        .to_matrix()[:, 0]
    )
    puccsd_state = (
        PUCCSDAnsatz(active_occupation)
        .build_circuit("spin")
        .assign_parameters(
            {
                "spin_theta_0": single_theta,
                "spin_theta_1": pair_rotation,
            }
        )
        .to_matrix()[:, 0]
    )
    _, singlet_hamiltonian, _, _ = _independent_two_orbital_singlet_model(matrix)
    seniority_zero = singlet_hamiltonian[np.ix_((0, 2), (0, 2))]
    seniority_zero_energy = float(np.linalg.eigvalsh(seniority_zero)[0])

    assert _fixed_particle_ground_energy(matrix, active.num_electrons) == pytest.approx(
        casci_energy, abs=1e-8
    )
    assert np.vdot(active_hf_state, matrix @ active_hf_state).real == pytest.approx(
        result.hartree_fock_energy, abs=1e-9
    )
    assert active_uccsd.num_parameters() == 3
    np.testing.assert_allclose(uccsd_zero_state, active_hf_state, atol=1e-12)
    for basis, amplitude in enumerate(uccsd_random_state):
        if abs(amplitude) <= 1e-12:
            continue
        num_alpha = sum(bool(basis & (1 << mode)) for mode in (0, 2))
        num_beta = sum(bool(basis & (1 << mode)) for mode in (1, 3))
        assert (num_alpha, num_beta) == (1, 1)
    assert float(np.vdot(puccd_state, matrix @ puccd_state).real) == pytest.approx(
        seniority_zero_energy, abs=1e-10
    )
    assert float(np.vdot(puccsd_state, matrix @ puccsd_state).real) == pytest.approx(
        casci_energy, abs=1e-8
    )
    assert abs(np.vdot(expected_singlet, puccsd_state)) == pytest.approx(1.0, abs=1e-10)
