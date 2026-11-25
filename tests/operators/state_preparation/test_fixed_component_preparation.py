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

"""Fixed component inputs against analytic matrices and algorithm references."""

import math
from dataclasses import replace

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.circuit import StandardGate
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms import (
    CHSH,
    HHL,
    QAOA,
    AmplitudeEstimationProblem,
    FasterAmplitudeEstimation,
    Grover,
    GroverRequest,
    HamiltonianPhaseEstimation,
    HamiltonianPhaseEstimationRequest,
    IterativeAmplitudeEstimation,
    LinearSystemProblem,
    MaximumLikelihoodAmplitudeEstimation,
    PhaseEstimationProblem,
    QBayesian,
    QuantumKernelTrainer,
    QuantumTeleportation,
    StandardAmplitudeEstimation,
)
from cqlib_algorithms.error_mitigation import QemExecutor, zne_executor
from cqlib_algorithms.operators import (
    AngleEncoding,
    BooleanFunction,
    FidelityQuantumKernel,
    GeneralStateReflection,
    GradientDescent,
    GroverOperator,
    ParameterShiftGradient,
    PhaseOracle,
    RealAmplitudesAnsatz,
    StatevectorEstimator,
    StatevectorSampler,
    TrainableFidelityQuantumKernel,
    UserCircuitPreparation,
)
from cqlib_algorithms.protocols import Sampler, StatePreparation


@pytest.mark.parametrize(
    "entry",
    [
        "reflection",
        "ae_preparation",
        "ae_iteration",
        "phase_preparation",
        "hamiltonian_preparation",
        "hhl_rhs",
        "bayesian_preparation",
        "chsh_build",
        "chsh_run",
        "teleportation_build",
        "teleportation_run",
        "fixed_kernel",
        "trainable_kernel",
        "kernel_trainer",
    ],
)
def test_role_specific_public_inputs_reject_bare_circuits(entry):
    circuit = Circuit(1)
    preparation = UserCircuitPreparation(circuit)
    oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [1]))
    iteration = GroverOperator(oracle, state_preparation=preparation)
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("Z"), 1.0)
    factories = {
        "reflection": lambda: GeneralStateReflection(circuit),
        "ae_preparation": lambda: AmplitudeEstimationProblem(
            circuit, iteration, (0,), bool
        ),
        "ae_iteration": lambda: AmplitudeEstimationProblem(
            preparation, circuit, (0,), bool
        ),
        "phase_preparation": lambda: PhaseEstimationProblem(circuit, circuit),
        "hamiltonian_preparation": lambda: HamiltonianPhaseEstimationRequest(
            hamiltonian, circuit, 2, 8
        ),
        "hhl_rhs": lambda: LinearSystemProblem(np.eye(2), circuit),
        "bayesian_preparation": lambda: QBayesian(
            circuit, {"x": 0}, StatevectorSampler()
        ),
        "chsh_build": lambda: CHSH(StatevectorEstimator()).build_circuit(circuit),
        "chsh_run": lambda: CHSH(StatevectorEstimator()).run(circuit),
        "teleportation_build": lambda: (
            QuantumTeleportation().build_premeasurement_circuit(circuit)
        ),
        "teleportation_run": lambda: QuantumTeleportation().run(circuit, shots=8),
        "fixed_kernel": lambda: FidelityQuantumKernel(circuit, ["x"]),
        "trainable_kernel": lambda: TrainableFidelityQuantumKernel(
            circuit, ["x"], ["w"]
        ),
        "kernel_trainer": lambda: QuantumKernelTrainer(circuit, ["x"], ["w"]),
    }
    with pytest.raises(
        TypeError, match="StatePreparation|component|GroverOperator|sequence"
    ):
        factories[entry]()


class _CachedPreparation(StatePreparation):
    def __init__(self, circuit, *, width=None, slots=None):
        self.circuit = circuit
        self.width = circuit.num_qubits if width is None else width
        self.slots = len(circuit.used_symbols) if slots is None else slots
        self.builds = 0

    def validate(self):
        pass

    def num_qubits(self):
        return self.width

    def num_parameters(self):
        return self.slots

    def build_circuit(self, prefix):
        self.builds += 1
        return self.circuit


@pytest.mark.parametrize("kind", ["amplitude", "phase", "hamiltonian", "linear"])
def test_problem_replacement_keeps_components_and_independent_execution_snapshots(kind):
    preparation = UserCircuitPreparation.from_component(
        AngleEncoding(1, StandardGate.RY), {"state_0": 0.43}, prefix="state"
    )
    unitary = Circuit(1)
    unitary.z(0)
    hamiltonian = Hamiltonian.from_pauli(PauliString.from_str("Z"))
    if kind == "amplitude":
        iteration = GroverOperator(
            PhaseOracle(BooleanFunction.from_marked_states(1, [1])),
            state_preparation=preparation,
        )
        original = AmplitudeEstimationProblem(preparation, iteration, (0,), bool)
    elif kind == "phase":
        original = PhaseEstimationProblem(unitary, preparation)
    elif kind == "hamiltonian":
        original = HamiltonianPhaseEstimationRequest(hamiltonian, preparation, 2, 8)
    else:
        original = LinearSystemProblem(np.eye(2), preparation)

    cloned = replace(original)
    cloned.validate()
    component_name = "rhs" if kind == "linear" else "state_preparation"
    circuit_name = "rhs_circuit" if kind == "linear" else "preparation_circuit"
    assert isinstance(getattr(cloned, component_name), StatePreparation)
    first, second = getattr(original, circuit_name), getattr(cloned, circuit_name)
    assert first is not second
    np.testing.assert_allclose(
        first.to_matrix(), second.to_matrix(), atol=1e-12, rtol=0
    )
    first.x(0)
    np.testing.assert_allclose(
        second.to_matrix(),
        preparation.build_circuit("").to_matrix(),
        atol=1e-12,
        rtol=0,
    )


class _SeededSampler(Sampler):
    def __init__(self):
        self.offset = 0

    def run(self, requests):
        batch = []
        for request in requests:
            batch.append(replace(request, seed=7 + self.offset))
            self.offset += 1
        return StatevectorSampler().run(batch)


@pytest.mark.parametrize("kind", ["preparation", "ansatz"])
def test_factory_binds_both_component_roles_to_the_analytic_rotation(kind):
    component = (
        AngleEncoding(1, StandardGate.RY)
        if kind == "preparation"
        else RealAmplitudesAnsatz(1, reps=0)
    )
    fixed = UserCircuitPreparation.from_component(
        component, {"features_0": 0.4}, prefix="features"
    )
    expected = np.array(
        [[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]]
    )
    assert isinstance(fixed, StatePreparation)
    assert fixed.num_parameters() == 0
    fixed.validate()
    for prefix in ("other", "", "features"):
        built = fixed.build_circuit(prefix)
        assert built.used_symbols == []
        np.testing.assert_allclose(built.to_matrix(), expected, atol=1e-14)


def test_binding_preserves_insertion_order_global_phase_and_snapshot_ownership():
    source = Circuit([7, 2])
    source.ry(7, Parameter("angle"))
    source.rz(2, Parameter("z"))
    source.set_global_phase(Parameter("phase"))
    component = _CachedPreparation(source)
    point = {"angle": 0.4, "z": -0.7, "phase": 0.3}
    fixed = UserCircuitPreparation.from_component(component, point)
    ry = np.array([[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]])
    rz = np.diag([np.exp(0.35j), np.exp(-0.35j)])
    expected = np.exp(0.3j) * np.kron(rz, ry)
    point["angle"] = 2.0
    source.x(7)
    first = fixed.build_circuit("one")
    assert [q.index for q in first.qubits] == [0, 1]
    np.testing.assert_allclose(first.to_matrix(), expected, atol=1e-14)
    first.x(0)
    np.testing.assert_allclose(
        fixed.build_circuit("two").to_matrix(), expected, atol=1e-14
    )
    assert component.builds == 1
    assert set(source.used_symbols) == {"angle", "z", "phase"}


def test_unused_logical_slots_do_not_require_invented_parameter_values():
    fixed = UserCircuitPreparation.from_component(
        _CachedPreparation(Circuit(1), slots=3), {}
    )
    assert fixed.num_parameters() == 0
    np.testing.assert_array_equal(fixed.build_circuit("unused").to_matrix(), np.eye(2))


@pytest.mark.parametrize(
    "bindings",
    [
        {},
        {"theta_0": 0.4, "extra": 1.0},
        {"theta_0": math.nan},
        {"theta_0": math.inf},
        {"theta_0": True},
        {0: 0.4},
        None,
        [0.4],
    ],
)
def test_fixed_component_rejects_invalid_bindings(bindings):
    with pytest.raises((TypeError, ValueError)):
        UserCircuitPreparation.from_component(RealAmplitudesAnsatz(1, reps=0), bindings)


@pytest.mark.parametrize("kind", ["width", "slots", "return", "zero", "boolean"])
def test_invalid_component_metadata_is_rejected(kind):
    source = Circuit(1)
    source.ry(0, Parameter("angle"))
    component = _CachedPreparation(
        source,
        width=2
        if kind == "width"
        else 0
        if kind == "zero"
        else True
        if kind == "boolean"
        else 1,
        slots=0 if kind == "slots" else 1,
    )
    if kind == "return":
        component.circuit = object()
    with pytest.raises((TypeError, ValueError)):
        UserCircuitPreparation.from_component(component, {"angle": 0.4})


@pytest.mark.parametrize("kind", ["measure", "reset", "delay"])
def test_invalid_preparation_operations_are_rejected_before_wrapping(kind):
    source = Circuit(1)
    if kind == "measure":
        source.measure(0)
    elif kind == "reset":
        source.reset(0)
    elif kind == "delay":
        source.delay(0, 1.0)
    with pytest.raises(ValueError):
        UserCircuitPreparation.from_component(_CachedPreparation(source), {})


def test_wrong_component_or_prefix_is_rejected():
    with pytest.raises(TypeError, match="component"):
        UserCircuitPreparation.from_component(object(), {})
    with pytest.raises(TypeError, match="component"):
        UserCircuitPreparation.from_component(Circuit(1), {})
    with pytest.raises(TypeError, match="prefix"):
        UserCircuitPreparation.from_component(
            RealAmplitudesAnsatz(1, reps=0), {}, prefix=2
        )


def test_general_reflection_direct_ansatz_and_fixed_component_match_analytic_projector():
    ansatz = RealAmplitudesAnsatz(1, reps=0)
    fixed = UserCircuitPreparation.from_component(ansatz, {"theta_0": 0.4})
    state = np.array([math.cos(0.2), math.sin(0.2)])
    expected = 2 * np.outer(state, state) - np.eye(2)
    symbolic = GeneralStateReflection(ansatz, prefix="alpha").build_circuit()
    np.testing.assert_allclose(
        symbolic.assign_parameters({"alpha_0": 0.4}).to_matrix(), expected, atol=1e-14
    )
    np.testing.assert_allclose(
        GeneralStateReflection(fixed).build_circuit().to_matrix(), expected, atol=1e-14
    )
    legacy = GeneralStateReflection.from_state_preparation(fixed, prefix="legacy")
    np.testing.assert_allclose(legacy.build_circuit().to_matrix(), expected, atol=1e-14)


def test_fixed_ansatz_builds_complete_grover_without_caller_circuits():
    preparation = UserCircuitPreparation.from_component(
        RealAmplitudesAnsatz(2, reps=0),
        {"theta_0": math.pi / 2, "theta_1": math.pi / 2},
    )
    function = BooleanFunction.from_marked_states(2, [2])
    iteration = GroverOperator(PhaseOracle(function), state_preparation=preparation)
    result = Grover(StatevectorSampler(), iteration=iteration).run(
        iterations=1, shots=64
    )
    legacy = Grover(StatevectorSampler()).run(
        [GroverRequest(function, shots=64, iterations=1)]
    )[0]
    assert result.counts == legacy.counts == (0, 0, 64, 0)


@pytest.mark.parametrize("method", ["ae", "iae", "mlae", "fae"])
def test_all_amplitude_estimators_reuse_the_bound_preparation(method):
    component = AngleEncoding(1, StandardGate.RY)
    fixed = UserCircuitPreparation.from_component(
        component, {"data_0": 0.4}, prefix="data"
    )
    step = GroverOperator(
        PhaseOracle(BooleanFunction.from_marked_states(1, [1])), state_preparation=fixed
    )
    problem = AmplitudeEstimationProblem(fixed, step, (0,), lambda bits: bits == "1")
    sampler = _SeededSampler()
    if method == "ae":
        result = StandardAmplitudeEstimation(sampler).run(
            problem, num_evaluation_qubits=6, shots=256
        )
    elif method == "iae":
        result = IterativeAmplitudeEstimation(sampler).run(
            problem, epsilon=0.005, alpha=0.05, shots=256
        )
    elif method == "mlae":
        result = MaximumLikelihoodAmplitudeEstimation(sampler).run(
            problem, powers=(0, 1, 2), shots=2048
        )
    else:
        result = FasterAmplitudeEstimation(sampler).run(
            problem, alpha=0.05, max_iterations=5
        )
    # sin(0.2) is below FAE's promised maximum amplitude of 0.25.
    assert result.estimation == pytest.approx(math.sin(0.2) ** 2, abs=0.008)


def test_fixed_input_is_shared_by_hpe_bayesian_and_mitigation_workflows():
    fixed = UserCircuitPreparation.from_component(
        RealAmplitudesAnsatz(1, reps=0), {"theta_0": math.pi}
    )
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("Z"), 1.0)
    energy = HamiltonianPhaseEstimation(StatevectorSampler()).run(
        hamiltonian,
        state_preparation=fixed,
        num_evaluation_qubits=3,
        shots=32,
        bound=2.0,
    )
    assert energy.eigenvalue == -1.0
    posterior = QBayesian(fixed, {"x": 0}, StatevectorSampler()).run(
        ["x"], target_accepted=32, shots_per_round=32
    )
    assert posterior.probabilities == {(1,): 1.0}
    mitigated = zne_executor(
        QemExecutor(estimator=StatevectorEstimator()), scale_factors=(1, 3, 5)
    ).run(fixed, PauliString.from_str("Z"))
    assert mitigated.expectation == pytest.approx(-1.0, abs=1e-12)


def test_fixed_component_is_a_linear_system_rhs_without_caller_circuits():
    rhs = UserCircuitPreparation.from_component(
        RealAmplitudesAnsatz(1, reps=0), {"theta_0": 0.0}
    )
    result = HHL(StatevectorSampler()).run(
        LinearSystemProblem(np.diag([1.0, 2.0]), rhs),
        num_evaluation_qubits=3,
        scaling=0.125,
        shots=256,
        bound=4.0,
        seed=7,
    )
    assert result.solution_probabilities == {0: 1.0}
    assert result.success_probability == pytest.approx(0.25, abs=0.08)


def test_fixed_component_satisfies_qaoas_zero_parameter_initial_state_contract():
    initial = UserCircuitPreparation.from_component(
        RealAmplitudesAnsatz(1, reps=0), {"theta_0": math.pi}
    )
    cost = Hamiltonian(1)
    cost.add_term(PauliString.from_str("Z"), 1.0)
    result = QAOA(
        StatevectorEstimator(),
        GradientDescent(ParameterShiftGradient(), maxiter=0),
        initial_state=initial,
    ).run(cost, {"qaoa_0_gamma": 0.3, "qaoa_0_beta": 0.2})
    assert result.optimal_value == pytest.approx(-math.cos(0.4), abs=1e-12)
