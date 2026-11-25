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

"""Scientific, execution-contract, and boundary tests for electronic SQD."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.device import ExecutionResult
from scipy.sparse.linalg import ArpackError

from cqlib_algorithms import (
    SQD,
    SQDIteration,
    SQDResult,
    SQDSubspaceResult,
    recover_sqd_configurations,
    solve_sqd_subspace,
)
from cqlib_algorithms.algorithms.electronic_structure.sqd import algorithm as sqd_module
from cqlib_algorithms.algorithms.electronic_structure.sqd import (
    subspace as sqd_subspace_module,
)
from cqlib_algorithms.extension.electronic_structure import (
    AngularMomentum,
    ElectronicEnergy,
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    SpinResolvedElectronicIntegrals,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import (
    ArbitraryStatePreparation,
    BasisStateInitialization,
    HartreeFockStatePreparation,
    UserCircuitPreparation,
)
from cqlib_algorithms.protocols import Sampler, SamplerRequest, StatePreparation


def test_component_run_matches_circuit_entry_and_independent_eigenproblem() -> None:
    system = _one_electron_system()
    amplitudes = np.zeros(16, dtype=complex)
    amplitudes[[1, 4]] = 1 / np.sqrt(2)
    preparation = ArbitraryStatePreparation(amplitudes.tolist())
    solved: list[tuple[int, ...]] = []

    def solver(selected_system, configurations):
        assert selected_system is system
        solved.append(configurations)
        return solve_sqd_subspace(selected_system, configurations)

    algorithm = SQD(StatevectorSampler(), subspace_solver=solver)
    options = {
        "shots": 128,
        "samples_per_batch": 2,
        "num_batches": 2,
        "max_iterations": 3,
        "seed": 19,
    }
    result = algorithm.run(system, preparation, **options)
    legacy = algorithm.compute(system, preparation, **options)
    assert result == legacy
    assert result.energy == pytest.approx(
        0.2 + np.linalg.eigvalsh([[-1.0, -0.3], [-0.3, 0.5]])[0], abs=1e-12
    )
    assert all(configurations == (1, 4) for configurations in solved)
    assert result.valid_raw_shots == 128
    assert result.converged


@pytest.mark.parametrize("index, energy", [(1, -0.8), (4, 0.7)])
def test_component_run_executes_the_selected_preparation(index, energy) -> None:
    result = SQD(StatevectorSampler()).run(
        _one_electron_system(),
        BasisStateInitialization(4, index),
        shots=16,
        samples_per_batch=1,
        seed=7,
    )
    assert result.ground_state.configurations == (index,)
    assert result.energy == pytest.approx(energy, abs=1e-12)


class _SQDPreparation(StatePreparation):
    def __init__(self, circuit, *, width=4, parameters=0):
        self.circuit = circuit
        self.width = width
        self.parameters = parameters

    def validate(self):
        pass

    def num_qubits(self):
        return self.width

    def num_parameters(self):
        return self.parameters

    def build_circuit(self, prefix):
        assert prefix == "sqd"
        return self.circuit


def test_component_run_snapshots_preparation_before_backend_execution() -> None:
    class MutatingSampler(Sampler):
        def run(self, requests):
            results = StatevectorSampler().run(requests)
            for request in requests:
                request.circuit.x(0)
            return results

    circuit = Circuit(4)
    circuit.x(0)
    original = circuit.to_matrix().copy()
    source = _SQDPreparation(circuit)
    algorithm = SQD(MutatingSampler())
    results = [
        algorithm.run(
            _one_electron_system(), source, shots=16, samples_per_batch=1, seed=7
        )
        for _ in range(2)
    ]
    assert results[0] == results[1]
    np.testing.assert_array_equal(circuit.to_matrix(), original)


@pytest.mark.parametrize(
    "width, parameters, output, error, message",
    [
        (True, 0, "bound", TypeError, "num_qubits"),
        (0, 0, "bound", ValueError, "num_qubits"),
        (4, -1, "bound", ValueError, "num_parameters"),
        (4, True, "bound", TypeError, "num_parameters"),
        (3, 0, "bound", ValueError, "qubit count"),
        (4, 0, "object", TypeError, "return a Circuit"),
        (4, 1, "symbolic", ValueError, "fully bound"),
        (4, 0, "reset", ValueError, "unitary"),
        (4, 0, "measurement", ValueError, "classical data"),
        (4, 0, "wrong_width", ValueError, "qubit count"),
        (3, 0, "wrong_width", ValueError, "width"),
    ],
)
def test_component_run_rejects_invalid_preparations_before_sampling(
    width, parameters, output, error, message
) -> None:
    circuit = Circuit(3 if output == "wrong_width" else 4)
    if output == "symbolic":
        circuit.rx(0, Parameter("theta"))
    elif output == "reset":
        circuit.reset(0)
    elif output == "measurement":
        circuit.measure(0)
    component = _SQDPreparation(
        object() if output == "object" else circuit,
        width=width,
        parameters=parameters,
    )
    sampler = _ScriptedSampler({"0001": 16})
    with pytest.raises(error, match=message):
        SQD(sampler).run(
            _one_electron_system(), component, shots=16, samples_per_batch=1
        )
    assert sampler.calls == []


def test_sqd_internal_modules_follow_one_way_dependency_order() -> None:
    base = Path(sqd_module.__file__).parent
    allowed: dict[str, set[str]] = {
        "results": set(),
        "recovery": {"results"},
        "subspace": {"recovery", "results"},
        "algorithm": {"recovery", "results", "subspace"},
    }
    for module_name, dependencies in allowed.items():
        tree = ast.parse((base / f"{module_name}.py").read_text(encoding="utf-8"))
        internal_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module in allowed
        }
        assert internal_imports <= dependencies


class _ScriptedSampler(Sampler):
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.calls: list[tuple[SamplerRequest, ...]] = []

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        batch = tuple(requests)
        self.calls.append(batch)
        return [
            ExecutionResult.from_counts(
                "sqd-test",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                dict(self.counts),
                "test-backend",
            )
            for request in batch
        ]


def _one_electron_system() -> ElectronicSystem:
    return ElectronicSystem(
        ElectronicIntegrals(
            0.2,
            np.array([[-1.0, -0.3], [-0.3, 0.5]]),
            np.zeros((2, 2, 2, 2)),
        ),
        ElectronicOccupation(2, 1, 0),
    )


def _closed_shell_system() -> ElectronicSystem:
    two_body = np.zeros((2, 2, 2, 2))
    for index in (
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
    ):
        two_body[index] = 0.2
    one_body = np.diag([-0.5, 0.5])
    one_body -= 2.0 * two_body[:, :, 0, 0]
    one_body += two_body[:, 0, 0, :]
    return ElectronicSystem(
        ElectronicIntegrals(0.4, one_body, two_body),
        ElectronicOccupation.closed_shell(2, 2),
    )


def _h2_sto3g_system() -> ElectronicSystem:
    """Return H2/STO-3G integrals from Whitfield et al., arXiv:1001.3855."""
    two_body = np.zeros((2, 2, 2, 2))

    def assign(value: float, p: int, r: int, q: int, s: int) -> None:
        for indices in {
            (p, r, q, s),
            (r, p, q, s),
            (p, r, s, q),
            (r, p, s, q),
            (q, s, p, r),
            (s, q, p, r),
            (q, s, r, p),
            (s, q, r, p),
        }:
            two_body[indices] = value

    assign(0.674493, 0, 0, 0, 0)
    assign(0.697397, 1, 1, 1, 1)
    assign(0.663472, 0, 0, 1, 1)
    assign(0.181287, 0, 1, 0, 1)
    return ElectronicSystem(
        ElectronicIntegrals(
            1.0 / 1.401,
            np.diag([-1.252477, -0.475934]),
            two_body,
        ),
        ElectronicOccupation.closed_shell(2, 2),
    )


def _compute(
    counts: dict[str, int],
    *,
    system: ElectronicSystem | None = None,
    max_iterations: int = 2,
    **kwargs,
) -> SQDResult:
    selected = _one_electron_system() if system is None else system
    shots = sum(counts.values())
    return SQD(_ScriptedSampler(counts)).compute(
        selected,
        UserCircuitPreparation(Circuit(selected.num_spin_orbitals)),
        shots=shots,
        samples_per_batch=shots,
        max_iterations=max_iterations,
        seed=17,
        **kwargs,
    )


@pytest.mark.reference
def test_one_electron_sqd_matches_independent_spatial_orbital_eigenproblem() -> None:
    system = _one_electron_system()
    result = _compute({"0001": 50, "0100": 50}, system=system)

    reference_values, reference_vectors = np.linalg.eigh(system.integrals.one_body)
    expected_energy = system.integrals.constant + reference_values[0]
    expected_occupancies = (
        abs(reference_vectors[0, 0]) ** 2,
        0.0,
        abs(reference_vectors[1, 0]) ** 2,
        0.0,
    )
    assert result.energy == pytest.approx(expected_energy, abs=1e-12)
    assert result.ground_state.configurations == (0b0001, 0b0100)
    assert result.orbital_occupancies == pytest.approx(
        expected_occupancies,
        abs=1e-12,
    )
    assert result.converged
    assert result.num_iterations == 2


@pytest.mark.reference
def test_two_electron_interacting_projection_matches_dense_fci_subblock() -> None:
    system = _closed_shell_system()
    result = _compute(
        {"0011": 40, "1100": 40},
        system=system,
        num_batches=3,
    )

    hamiltonian = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    expected = float(
        np.linalg.eigvalsh(hamiltonian[np.ix_((0b0011, 0b1100), (0b0011, 0b1100))])[0]
    )
    assert result.energy == pytest.approx(expected, abs=1e-12)
    assert all(len(iteration.subspaces) == 3 for iteration in result.iterations)
    assert all(
        subspace.energy + 1e-12 >= float(np.linalg.eigvalsh(hamiltonian)[0])
        for iteration in result.iterations
        for subspace in iteration.subspaces
    )


@pytest.mark.reference
def test_h2_sto3g_full_sector_matches_independent_dense_fci() -> None:
    system = _h2_sto3g_system()
    configurations = (0b0011, 0b0110, 0b1001, 0b1100)
    counts = {format(configuration, "04b"): 16 for configuration in configurations}

    result = SQD().compute_from_counts(
        system,
        counts,
        samples_per_batch=len(configurations),
        max_iterations=1,
        seed=11,
    )

    matrix = ElectronicEnergy(system.integrals).to_qubit_hamiltonian().to_matrix()
    expected = float(
        np.linalg.eigvalsh(matrix[np.ix_(configurations, configurations)])[0]
    )
    assert result.ground_state.configurations == configurations
    assert result.energy == pytest.approx(expected, abs=1e-12)
    assert result.energy == pytest.approx(-1.137269804069418, abs=1e-12)


@pytest.mark.reference
def test_spin_symmetrization_closes_open_shell_partners_and_spans_s2_sectors() -> None:
    system = _closed_shell_system()
    result = SQD().compute_from_counts(
        system,
        {"1001": 32},
        samples_per_batch=4,
        max_iterations=1,
        symmetrize_spin=True,
        seed=4,
    )

    configurations = result.ground_state.configurations
    assert configurations == (0b0011, 0b0110, 0b1001, 0b1100)
    spin_squared = AngularMomentum(2).to_qubit_hamiltonian().to_matrix()
    projected = spin_squared[np.ix_(configurations, configurations)]
    np.testing.assert_allclose(
        np.linalg.eigvalsh(projected),
        np.array([0.0, 0.0, 0.0, 2.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_every_spin_symmetrized_batch_is_a_bounded_cartesian_closure() -> None:
    num_spatial_orbitals = 3
    system = ElectronicSystem(
        ElectronicIntegrals(
            0.0,
            np.zeros((num_spatial_orbitals,) * 2),
            np.zeros((num_spatial_orbitals,) * 4),
        ),
        ElectronicOccupation.closed_shell(num_spatial_orbitals, 2),
    )
    result = SQD().compute_from_counts(
        system,
        {"000011": 5, "100100": 3, "011000": 2},
        samples_per_batch=9,
        num_batches=5,
        max_iterations=1,
        symmetrize_spin=True,
        seed=14,
    )

    for subspace in result.iterations[0].subspaces:
        alpha = {
            sum(
                ((configuration >> (2 * orbital)) & 1) << orbital
                for orbital in range(num_spatial_orbitals)
            )
            for configuration in subspace.configurations
        }
        beta = {
            sum(
                ((configuration >> (2 * orbital + 1)) & 1) << orbital
                for orbital in range(num_spatial_orbitals)
            )
            for configuration in subspace.configurations
        }
        assert alpha == beta
        assert subspace.dimension == len(alpha) ** 2
        assert subspace.dimension <= 9


@pytest.mark.reference
def test_configuration_recovery_repairs_each_spin_sector() -> None:
    system = ElectronicSystem(
        ElectronicIntegrals(0.0, np.array([[-1.0]]), np.zeros((1, 1, 1, 1))),
        ElectronicOccupation(1, 1, 0),
    )
    result = _compute(
        {"01": 2, "10": 3, "11": 5},
        system=system,
        max_iterations=1,
    )

    assert result.valid_raw_shots == 2
    assert result.iterations[0].num_recovered_configurations == 1
    assert result.ground_state.configurations == (0b01,)
    assert result.orbital_occupancies == pytest.approx((1.0, 0.0), abs=1e-12)


def test_setup_round_postselects_the_valid_raw_sector() -> None:
    system = _one_electron_system()
    # With no initial occupancies, max_iterations=1 executes only the setup
    # diagonalization. Invalid all-zero outcomes must not enter its subspace.
    result = _compute(
        {"0100": 1, "0000": 99},
        system=system,
        max_iterations=1,
    )

    assert result.valid_raw_shots == 1
    assert result.ground_state.configurations == (0b0100,)
    assert result.energy == pytest.approx(0.7, abs=1e-12)


def test_setup_diagonalization_precedes_recovery_and_supplies_ritz_occupancies(
    monkeypatch,
) -> None:
    system = ElectronicSystem(
        ElectronicIntegrals(
            0.0,
            np.diag([-1.0, 1.0]),
            np.zeros((2,) * 4),
        ),
        ElectronicOccupation(2, 1, 0),
    )
    events: list[tuple[str, object]] = []
    original_recovery = sqd_module._recover_distribution

    def tracked_recovery(distribution, selected_system, occupancies, rng):
        events.append(("recover", occupancies))
        return original_recovery(distribution, selected_system, occupancies, rng)

    def tracked_solver(selected_system, configurations):
        events.append(("solve", configurations))
        return solve_sqd_subspace(selected_system, configurations)

    monkeypatch.setattr(sqd_module, "_recover_distribution", tracked_recovery)
    result = SQD(subspace_solver=tracked_solver).compute_from_counts(
        system,
        {"0001": 1, "0100": 1, "0000": 100},
        samples_per_batch=2,
        max_iterations=2,
        carryover_threshold=None,
        seed=21,
    )

    assert tuple(event[0] for event in events) == ("solve", "recover", "solve")
    assert events[0][1] == (0b0001, 0b0100)
    assert events[1][1] == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-12)
    assert events[1][1] != pytest.approx((0.5, 0.0, 0.5, 0.0), abs=1e-12)
    assert result.iterations[0].orbital_occupancies == pytest.approx(
        (1.0, 0.0, 0.0, 0.0),
        abs=1e-12,
    )


def test_user_initial_occupancies_allow_recovery_without_valid_raw_shots() -> None:
    system = _one_electron_system()
    result = SQD(_ScriptedSampler({"0000": 8})).compute(
        system,
        UserCircuitPreparation(Circuit(4)),
        shots=8,
        samples_per_batch=8,
        max_iterations=1,
        initial_occupancies=(1.0, 0.0, 0.0, 0.0),
        seed=5,
    )

    assert result.valid_raw_shots == 0
    assert result.ground_state.configurations == (0b0001,)
    assert result.energy == pytest.approx(-0.8, abs=1e-12)


def test_missing_valid_sector_and_initial_occupancies_is_rejected() -> None:
    with pytest.raises(ValueError, match="no target-sector"):
        _compute({"0000": 4}, max_iterations=1)


def test_selected_subspace_energy_is_variational_and_improves_with_support() -> None:
    system = _one_electron_system()
    restricted = _compute(
        {"0001": 10},
        system=system,
        max_iterations=1,
    )
    expanded = _compute(
        {"0001": 5, "0100": 5},
        system=system,
        max_iterations=1,
    )

    exact = system.integrals.constant + float(
        np.linalg.eigvalsh(system.integrals.one_body)[0]
    )
    assert restricted.energy >= expanded.energy - 1e-12
    assert expanded.energy == pytest.approx(exact, abs=1e-12)


def test_existing_counts_can_run_without_a_sampler() -> None:
    system = _one_electron_system()
    counts = {"0001": 5, "0100": 5}

    from_counts = SQD().compute_from_counts(
        system,
        counts,
        samples_per_batch=2,
        max_iterations=2,
        seed=13,
    )
    from_sampler = _compute(counts, system=system)

    assert from_counts.energy == pytest.approx(from_sampler.energy, abs=1e-12)
    assert from_counts.ground_state.configurations == (0b0001, 0b0100)
    assert from_counts.shots == 10


def test_included_configuration_is_present_when_it_was_not_sampled() -> None:
    system = _one_electron_system()
    result = SQD().compute_from_counts(
        system,
        {"0100": 8},
        samples_per_batch=2,
        max_iterations=1,
        include_configurations=(0b0001,),
        seed=2,
    )

    assert result.ground_state.configurations == (0b0001, 0b0100)
    exact = system.integrals.constant + float(
        np.linalg.eigvalsh(system.integrals.one_body)[0]
    )
    assert result.energy == pytest.approx(exact, abs=1e-12)


def test_high_amplitude_configuration_is_carried_into_the_next_iteration() -> None:
    result = SQD().compute_from_counts(
        _one_electron_system(),
        {"0001": 4, "0100": 4},
        samples_per_batch=1,
        max_iterations=2,
        energy_tolerance=0.0,
        occupancy_tolerance=0.0,
        carryover_threshold=0.0,
        seed=8,
    )

    assert result.num_iterations == 2
    assert (
        result.iterations[1].subspaces[0].configurations
        == result.iterations[0].best_subspace.configurations
    )


def test_carryover_capacity_prefers_largest_absolute_amplitudes() -> None:
    amplitudes = (0.1, 0.8, 0.5, math.sqrt(0.1))
    result = SQDSubspaceResult(
        -1.0,
        (0b0011, 0b0110, 0b1001, 0b1100),
        tuple(complex(value) for value in amplitudes),
        4,
    )

    selected = sqd_subspace_module._select_required_configurations(
        (),
        result,
        carryover_threshold=0.0,
        samples_per_batch=2,
        symmetrize_spin=False,
        num_spatial_orbitals=2,
    )

    assert selected == (0b0110, 0b1001)


def test_callback_observes_each_completed_iteration() -> None:
    observed: list[SQDIteration] = []
    result = SQD().compute_from_counts(
        _one_electron_system(),
        {"0001": 4, "0100": 4},
        samples_per_batch=2,
        max_iterations=2,
        callback=observed.append,
        seed=3,
    )

    assert observed == list(result.iterations)


def test_custom_subspace_solver_can_replace_the_reference_solver() -> None:
    calls: list[tuple[int, ...]] = []

    def solver(
        system: ElectronicSystem,
        configurations: tuple[int, ...],
    ) -> SQDSubspaceResult:
        calls.append(configurations)
        return solve_sqd_subspace(system, configurations)

    result = SQD(subspace_solver=solver).compute_from_counts(
        _one_electron_system(),
        {"0001": 4, "0100": 4},
        samples_per_batch=2,
        max_iterations=1,
        seed=9,
    )

    assert calls == [(0b0001, 0b0100)]
    assert result.energy == pytest.approx(
        solve_sqd_subspace(_one_electron_system(), (0b0001, 0b0100)).energy,
        abs=1e-12,
    )


def test_custom_subspace_solver_response_is_strictly_validated() -> None:
    def wrong_type(system, configurations):
        return object()

    with pytest.raises(TypeError, match="SQDSubspaceResult"):
        SQD(subspace_solver=wrong_type).compute_from_counts(
            _one_electron_system(),
            {"0001": 2},
            samples_per_batch=1,
            max_iterations=1,
        )

    def wrong_subspace(system, configurations):
        return SQDSubspaceResult(-1.0, (0b0100,), (1.0 + 0.0j,), 4)

    with pytest.raises(ValueError, match="requested configurations"):
        SQD(subspace_solver=wrong_subspace).compute_from_counts(
            _one_electron_system(),
            {"0001": 2},
            samples_per_batch=1,
            max_iterations=1,
        )

    def wrong_width(system, configurations):
        return SQDSubspaceResult(-1.0, configurations, (1.0 + 0.0j,), 2)

    with pytest.raises(ValueError, match="width"):
        SQD(subspace_solver=wrong_width).compute_from_counts(
            _one_electron_system(),
            {"0001": 2},
            samples_per_batch=1,
            max_iterations=1,
        )


def test_public_recovery_repairs_particle_numbers_reproducibly() -> None:
    system = _one_electron_system()
    occupancies = (1.0, 0.0, 0.0, 0.0)
    first = recover_sqd_configurations(
        {0b0000: 0.75, 0b0101: 0.25}, system, occupancies, seed=5
    )
    second = recover_sqd_configurations(
        {0b0000: 3.0, 0b0101: 1.0}, system, occupancies, seed=5
    )

    assert first == second
    assert math.fsum(first.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(configuration.bit_count() == 1 for configuration in first)


@pytest.mark.parametrize(
    ("distribution", "error", "message"),
    [
        ([(0b0001, 1.0)], TypeError, "mapping"),
        ({True: 1.0}, TypeError, "integers"),
        ({1 << 4: 1.0}, ValueError, "outside"),
        ({0b0001: "1"}, TypeError, "real"),
        ({0b0001: -1.0}, ValueError, "non-negative"),
        ({0b0001: math.nan}, ValueError, "finite"),
        ({0b0001: 0.0}, ValueError, "positive total"),
    ],
)
def test_public_recovery_rejects_invalid_distributions(
    distribution,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        recover_sqd_configurations(
            distribution,
            _one_electron_system(),
            (1.0, 0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("configurations", "error", "message"),
    [
        ("0001", TypeError, "sequence"),
        ((), ValueError, "empty"),
        ((0b0001, 0b0001), ValueError, "unique"),
        ((1 << 4,), ValueError, "outside"),
        ((0b0011,), ValueError, "particle sector"),
    ],
)
def test_public_subspace_solver_rejects_invalid_configurations(
    configurations,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        solve_sqd_subspace(_one_electron_system(), configurations)


def test_public_subspace_solver_rejects_an_invalid_system() -> None:
    with pytest.raises(TypeError, match="ElectronicSystem"):
        solve_sqd_subspace(object(), (0b0001,))  # type: ignore[arg-type]


def test_public_subspace_solver_propagates_sparse_eigensolver_failure(
    monkeypatch,
) -> None:
    num_orbitals = 8
    system = ElectronicSystem(
        ElectronicIntegrals(
            0.0,
            np.diag(np.arange(1, num_orbitals + 1, dtype=np.float64)),
            np.zeros((num_orbitals,) * 4),
        ),
        ElectronicOccupation(num_orbitals, 4, 0),
    )
    configurations = tuple(
        sum(1 << (2 * orbital) for orbital in occupied)
        for occupied in combinations(range(num_orbitals), 4)
    )
    assert len(configurations) > sqd_subspace_module._DENSE_SOLVER_CUTOFF

    def failed_eigensolver(*args, **kwargs):
        raise ArpackError(-1)

    monkeypatch.setattr(sqd_subspace_module, "eigsh", failed_eigensolver)
    with pytest.raises(ArpackError, match="ARPACK error"):
        solve_sqd_subspace(system, configurations)


def test_ground_state_properties_correspond_to_the_reported_ritz_state() -> None:
    result = _compute(
        {"0001": 5, "0100": 5},
        system=_one_electron_system(),
        max_iterations=2,
    )

    assert result.ground_state_occupancies == result.ground_state.orbital_occupancies
    assert result.best_iteration_index in range(result.num_iterations)


def test_seed_reproduces_recovery_batches_and_sampler_seed() -> None:
    counts = {"0001": 4, "0100": 4, "0000": 4, "0101": 4}
    first_sampler = _ScriptedSampler(counts)
    second_sampler = _ScriptedSampler(counts)
    first = SQD(first_sampler).compute(
        _one_electron_system(),
        UserCircuitPreparation(Circuit(4)),
        shots=16,
        samples_per_batch=1,
        num_batches=4,
        max_iterations=3,
        seed=123,
    )
    second = SQD(second_sampler).compute(
        _one_electron_system(),
        UserCircuitPreparation(Circuit(4)),
        shots=16,
        samples_per_batch=1,
        num_batches=4,
        max_iterations=3,
        seed=123,
    )

    assert first == second
    assert first_sampler.calls[0][0].seed == second_sampler.calls[0][0].seed
    assert first_sampler.calls[0][0].seed != 123


@pytest.mark.integration
def test_statevector_sampler_and_hartree_fock_preparation_integrate() -> None:
    system = ElectronicSystem(
        ElectronicIntegrals(
            0.3,
            np.diag([-0.8, 0.4]),
            np.zeros((2, 2, 2, 2)),
        ),
        ElectronicOccupation(2, 1, 0),
    )
    circuit = HartreeFockStatePreparation(system.occupation).build_circuit("")
    before = len(circuit.operations)

    result = SQD(StatevectorSampler()).compute(
        system,
        UserCircuitPreparation(circuit),
        shots=32,
        samples_per_batch=8,
        max_iterations=2,
        seed=71,
    )

    assert result.energy == pytest.approx(-0.5, abs=1e-12)
    assert result.ground_state.configurations == (0b0001,)
    assert result.valid_raw_shots == 32
    assert len(circuit.operations) == before
    assert not circuit.classical_values


def test_noncontiguous_qubit_identifiers_preserve_position_semantics() -> None:
    system = _one_electron_system()
    circuit = Circuit([7, 2, 9, 4])
    sampler = _ScriptedSampler({"0001": 4, "0100": 4})

    result = SQD(sampler).run(
        system,
        _SQDPreparation(circuit),
        shots=8,
        samples_per_batch=2,
        max_iterations=1,
        seed=3,
    )

    assert sampler.calls[0][0].qubits == tuple(circuit.qubits)
    assert result.ground_state.configurations == (1, 4)


def test_spin_resolved_integrals_are_supported() -> None:
    integrals = SpinResolvedElectronicIntegrals(
        0.1,
        np.diag([-1.0, 0.2]),
        np.diag([-0.7, 0.4]),
        np.zeros((2,) * 4),
        np.zeros((2,) * 4),
        np.zeros((2,) * 4),
    )
    system = ElectronicSystem(
        integrals,
        ElectronicOccupation(2, 1, 1),
        alpha_beta_overlap=np.eye(2),
    )
    result = _compute({"0011": 8}, system=system, max_iterations=1)

    assert result.energy == pytest.approx(-1.6, abs=1e-12)


def test_zero_hamiltonian_has_a_deterministic_degenerate_ritz_state() -> None:
    system = ElectronicSystem(
        ElectronicIntegrals(0.0, np.zeros((2, 2)), np.zeros((2,) * 4)),
        ElectronicOccupation(2, 1, 0),
    )
    result = _compute(
        {"0001": 4, "0100": 4},
        system=system,
        max_iterations=1,
    )

    assert result.energy == 0.0
    assert result.ground_state.amplitudes == (1.0 + 0.0j, 0.0 + 0.0j)


@pytest.mark.reference
def test_sparse_ritz_path_matches_independent_diagonal_determinant_minimum() -> None:
    num_orbitals = 8
    num_alpha = 4
    energies = np.arange(num_orbitals, dtype=np.float64)
    system = ElectronicSystem(
        ElectronicIntegrals(
            0.25,
            np.diag(energies),
            np.zeros((num_orbitals,) * 4),
        ),
        ElectronicOccupation(num_orbitals, num_alpha, 0),
    )
    configurations = tuple(
        sum(1 << (2 * orbital) for orbital in occupied)
        for occupied in combinations(range(num_orbitals), num_alpha)
    )
    assert len(configurations) > 64
    counts = {
        format(configuration, f"0{2 * num_orbitals}b"): 1
        for configuration in configurations
    }

    result = _compute(counts, system=system, max_iterations=1)

    expected = 0.25 + float(np.sum(energies[:num_alpha]))
    assert result.energy == pytest.approx(expected, abs=1e-10)
    assert result.ground_state.dimension == len(configurations)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"shots": True}, TypeError, "shots"),
        ({"shots": 0}, ValueError, "shots"),
        ({"samples_per_batch": 0}, ValueError, "samples_per_batch"),
        ({"num_batches": 1.5}, TypeError, "num_batches"),
        ({"max_iterations": 0}, ValueError, "max_iterations"),
        ({"energy_tolerance": -1.0}, ValueError, "energy_tolerance"),
        ({"occupancy_tolerance": math.inf}, ValueError, "occupancy_tolerance"),
        ({"carryover_threshold": -1.0}, ValueError, "carryover_threshold"),
        ({"callback": object()}, TypeError, "callback"),
    ],
)
def test_numeric_options_are_validated_before_sampling(changes, error, message) -> None:
    sampler = _ScriptedSampler({"0001": 2})
    options = {
        "shots": 2,
        "samples_per_batch": 2,
        "num_batches": 1,
        "max_iterations": 1,
        "energy_tolerance": 1e-8,
        "occupancy_tolerance": 1e-5,
    }
    options.update(changes)
    with pytest.raises(error, match=message):
        SQD(sampler).compute(
            _one_electron_system(), UserCircuitPreparation(Circuit(4)), **options
        )
    assert sampler.calls == []


@pytest.mark.parametrize(
    ("counts", "error", "message"),
    [
        ({"001": 1}, ValueError, "width"),
        ({"00x1": 1}, ValueError, "width"),
        ({"0001": True}, TypeError, "integers"),
        ({"0001": -1}, ValueError, "non-negative"),
        ({"0001": 0}, ValueError, "positive total"),
    ],
)
def test_existing_counts_are_strictly_validated(counts, error, message) -> None:
    with pytest.raises(error, match=message):
        SQD().compute_from_counts(
            _one_electron_system(),
            counts,
            samples_per_batch=1,
        )


def test_required_configurations_are_validated_before_sampling() -> None:
    sampler = _ScriptedSampler({"0001": 2})
    with pytest.raises(ValueError, match="particle sector"):
        SQD(sampler).compute(
            _one_electron_system(),
            UserCircuitPreparation(Circuit(4)),
            shots=2,
            samples_per_batch=1,
            include_configurations=(0b0011,),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        SQD(sampler).compute(
            _one_electron_system(),
            UserCircuitPreparation(Circuit(4)),
            shots=2,
            samples_per_batch=1,
            include_configurations=(0b0001, 0b0100),
        )
    assert sampler.calls == []


@pytest.mark.parametrize(
    ("configurations", "message"),
    [
        ((0b0001, 0b0001), "unique"),
        ((1 << 4,), "outside"),
    ],
)
def test_required_configurations_reject_duplicates_and_out_of_range_values(
    configurations,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        SQD().compute_from_counts(
            _one_electron_system(),
            {"0001": 2},
            samples_per_batch=2,
            include_configurations=configurations,
        )


def test_counts_and_required_configuration_inputs_are_not_mutated() -> None:
    counts = {"0001": 2, "0100": 1}
    configurations = [0b0001]
    counts_before = dict(counts)
    configurations_before = list(configurations)

    SQD().compute_from_counts(
        _one_electron_system(),
        counts,
        samples_per_batch=2,
        max_iterations=1,
        include_configurations=configurations,
        seed=2,
    )

    assert counts == counts_before
    assert configurations == configurations_before


@pytest.mark.parametrize(
    ("system", "samples_per_batch", "symmetrize_spin", "error", "message"),
    [
        (_closed_shell_system(), 4, 1, TypeError, "bool"),
        (_one_electron_system(), 4, True, ValueError, "equal alpha and beta"),
        (_closed_shell_system(), 3, True, ValueError, "at least 4"),
    ],
)
def test_spin_symmetrization_contract_is_validated_before_sampling(
    system,
    samples_per_batch,
    symmetrize_spin,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        SQD().compute_from_counts(
            system,
            {"0011": 2},
            samples_per_batch=samples_per_batch,
            symmetrize_spin=symmetrize_spin,
        )


def test_spin_closure_of_required_configurations_respects_batch_capacity() -> None:
    system = ElectronicSystem(
        ElectronicIntegrals(0.0, np.zeros((3, 3)), np.zeros((3,) * 4)),
        ElectronicOccupation.closed_shell(3, 2),
    )
    with pytest.raises(ValueError, match="spin closure"):
        SQD().compute_from_counts(
            system,
            {"000011": 2},
            samples_per_batch=4,
            include_configurations=(0b000011, 0b011000),
            symmetrize_spin=True,
        )


@pytest.mark.parametrize(
    ("occupancies", "error", "message"),
    [
        ([1.0, 0.0, 0.0], ValueError, "length"),
        ([1.0, 0.0, 0.0, "x"], TypeError, "real"),
        ([1.1, 0.0, -0.1, 0.0], ValueError, "interval"),
        ([0.4, 0.0, 0.4, 0.0], ValueError, "num_alpha"),
        ([1.0, 0.1, 0.0, 0.0], ValueError, "num_beta"),
    ],
)
def test_initial_occupancies_are_physically_validated(
    occupancies, error, message
) -> None:
    sampler = _ScriptedSampler({"0000": 2})
    with pytest.raises(error, match=message):
        SQD(sampler).compute(
            _one_electron_system(),
            UserCircuitPreparation(Circuit(4)),
            shots=2,
            samples_per_batch=1,
            initial_occupancies=occupancies,
        )
    assert sampler.calls == []


def test_components_circuit_and_sampler_contract_are_validated() -> None:
    sampler = _ScriptedSampler({"0001": 2})
    with pytest.raises(TypeError, match="Sampler"):
        SQD(object())  # type: ignore[arg-type]
    algorithm = SQD(sampler)
    with pytest.raises(TypeError, match="ElectronicSystem"):
        algorithm.compute(
            object(), UserCircuitPreparation(Circuit(4)), shots=2, samples_per_batch=1
        )  # type: ignore[arg-type]
    for unsupported in (object(), Circuit(4)):
        with pytest.raises(TypeError, match="StatePreparation"):
            algorithm.compute(
                _one_electron_system(), unsupported, shots=2, samples_per_batch=1
            )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        algorithm.compute(
            _one_electron_system(),
            UserCircuitPreparation(Circuit(3)),
            shots=2,
            samples_per_batch=1,
        )

    symbolic = Circuit(4)
    symbolic.rx(0, Parameter("theta"))
    with pytest.raises(ValueError, match="fully bound"):
        algorithm.compute(
            _one_electron_system(),
            UserCircuitPreparation(symbolic),
            shots=2,
            samples_per_batch=1,
        )
    dynamic = Circuit(4)
    dynamic.reset(0)
    with pytest.raises(ValueError, match="unitary"):
        algorithm.compute(
            _one_electron_system(),
            _SQDPreparation(dynamic),
            shots=2,
            samples_per_batch=1,
        )
    assert sampler.calls == []


def test_invalid_sampler_batch_is_rejected() -> None:
    class _MissingResultSampler(Sampler):
        def run(self, requests):
            return []

    with pytest.raises(ValueError, match="one result per request"):
        SQD(_MissingResultSampler()).compute(
            _one_electron_system(),
            UserCircuitPreparation(Circuit(4)),
            shots=2,
            samples_per_batch=1,
        )


def test_result_models_are_frozen_validated_and_concisely_represented() -> None:
    state = SQDSubspaceResult(-1.0, (1,), (1.0 + 0.0j,), 2)
    iteration = SQDIteration(0, (state,), (1.0, 0.0), 1)
    result = SQDResult(state, (1.0, 0.0), (iteration,), False, 4, 3)

    assert "amplitudes" not in repr(state)
    assert "energy=-1.0" in repr(iteration)
    assert "num_iterations=1" in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.converged = True  # type: ignore[misc]
    with pytest.raises(ValueError, match="normalized"):
        replace(state, amplitudes=(0.5 + 0.0j,))
    with pytest.raises(ValueError, match="sorted and unique"):
        SQDSubspaceResult(-1.0, (2, 1), (1 / math.sqrt(2),) * 2, 2)
    with pytest.raises(ValueError, match="consecutive"):
        replace(result, iterations=(replace(iteration, index=1),))


def test_full_register_support_uses_sparse_projection_not_dense_hilbert_matrix(
    monkeypatch,
) -> None:
    system = _one_electron_system()

    def forbidden(*args, **kwargs):
        pytest.fail("SQD must not call Hamiltonian.to_matrix")

    monkeypatch.setattr("cqlib.qis.Hamiltonian.to_matrix", forbidden)
    result = _compute(
        {"0001": 5, "0100": 5},
        system=system,
        max_iterations=1,
    )
    assert math.isfinite(result.energy)
