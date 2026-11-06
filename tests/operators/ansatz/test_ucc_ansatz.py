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

"""Independent matrix, integration, and boundary tests for ordered UCC."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import (
    FermionicOp,
    Hamiltonian,
    JordanWignerMapper,
    PauliString,
)

from cqlib_algorithms.extension.electronic_structure import FermionicExcitation
from cqlib_algorithms.operators.ansatz import UCCAnsatz
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.state_preparation import BasisStateInitialization
from cqlib_algorithms.protocols import (
    EstimatorRequest,
    GradientRequest,
    StatePreparation,
)


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _transition_matrix(excitation: FermionicExcitation) -> np.ndarray:
    size = 1 << excitation.num_spin_orbitals
    transition = np.zeros((size, size), dtype=np.complex128)
    operations = tuple((True, target) for target in excitation.targets) + tuple(
        (False, source) for source in reversed(excitation.sources)
    )
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
            transition[output, basis] = amplitude
    return transition


def _generator(excitation: FermionicExcitation) -> np.ndarray:
    transition = _transition_matrix(excitation)
    return 1j * (transition - transition.conj().T)


def _evolution(generator: np.ndarray, theta: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(generator)
    return (vectors * np.exp(-1j * theta * values)) @ vectors.conj().T


@pytest.mark.reference
@pytest.mark.parametrize(
    "excitation",
    [
        FermionicExcitation(4, (0,), (2,)),
        FermionicExcitation(4, (0, 1), (2, 3)),
        FermionicExcitation(8, (0, 2), (4, 6)),
        FermionicExcitation(8, (1, 3), (5, 7)),
    ],
)
def test_each_factor_matches_independent_fock_exponential(excitation):
    theta = 0.37
    actual = (
        UCCAnsatz((excitation,))
        .build_circuit("ucc")
        .assign_parameters({"ucc_theta_0": theta})
        .to_matrix()
    )
    expected = _evolution(_generator(excitation), theta)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.reference
def test_noncommuting_factors_use_input_time_order():
    earlier = FermionicExcitation(4, (0,), (2,))
    later = FermionicExcitation(4, (0, 1), (2, 3))
    first_generator = _generator(earlier)
    second_generator = _generator(later)
    commutator = first_generator @ second_generator - second_generator @ first_generator
    assert np.max(np.abs(commutator)) > 0.5

    first_theta = 0.31
    second_theta = -0.47
    actual = (
        UCCAnsatz((earlier, later))
        .build_circuit("u")
        .assign_parameters({"u_theta_0": first_theta, "u_theta_1": second_theta})
        .to_matrix()
    )
    first = _evolution(first_generator, first_theta)
    second = _evolution(second_generator, second_theta)
    expected = second @ first
    reversed_order = first @ second
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert not np.allclose(actual, reversed_order, rtol=0.0, atol=1e-6)


def test_duplicate_factors_remain_ordered_independent_parameters():
    excitation = FermionicExcitation(4, (0,), (2,))
    component = UCCAnsatz((excitation, excitation))
    assert component.num_parameters() == 2
    assert component.build_circuit("x").used_symbols == [
        "x_theta_0",
        "x_theta_1",
    ]


def test_excitation_container_and_mapper_outputs_are_snapshotted():
    excitation = FermionicExcitation(4, (0,), (2,))
    excitation_list = [excitation]

    class RecordingMapper:
        def __init__(self) -> None:
            self.calls = 0
            self.outputs: list[Hamiltonian] = []

        def num_qubits(self, num_modes: int) -> int:
            return num_modes

        def map_operator(self, operator: FermionicOp) -> Hamiltonian:
            self.calls += 1
            output = JordanWignerMapper().map_operator(operator)
            self.outputs.append(output)
            return output

    mapper = RecordingMapper()
    component = UCCAnsatz(excitation_list, mapper=mapper)
    before = component.build_circuit("p").assign_parameters({"p_theta_0": 0.2})
    excitation_list.append(FermionicExcitation(4, (1,), (3,)))
    mapper.outputs[0].add_term(PauliString.from_str("XIII"), 9.0)
    after = component.build_circuit("p").assign_parameters({"p_theta_0": 0.2})

    assert mapper.calls == 1
    assert component.num_parameters() == 1
    np.testing.assert_allclose(after.to_matrix(), before.to_matrix(), atol=0.0)


class _FixedCircuitPreparation(StatePreparation):
    def __init__(
        self,
        circuit: object,
        *,
        width: int,
        parameters: int = 0,
    ) -> None:
        self.circuit = circuit
        self.width = width
        self.parameters = parameters
        self.calls = 0

    def validate(self) -> None:
        return None

    def build_circuit(self, prefix: str) -> Circuit:
        self.calls += 1
        return self.circuit  # type: ignore[return-value]

    def num_parameters(self) -> int:
        return self.parameters

    def num_qubits(self) -> int:
        return self.width


def test_fixed_initial_state_is_built_once_and_snapshotted():
    source = Circuit(4)
    source.x(1)
    preparation = _FixedCircuitPreparation(source, width=4)
    component = UCCAnsatz(
        (FermionicExcitation(4, (0,), (2,)),),
        initial_state=preparation,
    )
    source.x(0)
    first = component.build_circuit("p")
    second = component.build_circuit("p")
    first.x(3)

    assert preparation.calls == 1
    assert len(second.operations) + 1 == len(first.operations)
    zero = {"p_theta_0": 0.0}
    expected = BasisStateInitialization(4, 0b0010).build_circuit("").to_matrix()
    np.testing.assert_allclose(
        second.assign_parameters(zero).to_matrix(), expected, rtol=0.0, atol=1e-12
    )


def test_parameter_shift_matches_independent_central_difference():
    circuit = UCCAnsatz(
        (FermionicExcitation(4, (0,), (2,)),),
        initial_state=BasisStateInitialization(4, 1),
    ).build_circuit("u")
    observable = JordanWignerMapper().map_operator(FermionicOp.number(4, 0))
    binding = {"u_theta_0": 0.31}
    request = GradientRequest(
        EstimatorRequest(circuit, observable, binding),
        ("u_theta_0",),
    )
    actual = (
        ParameterShiftGradient().run(StatevectorEstimator(), [request])[0].values[0]
    )

    def expectation(theta: float) -> float:
        state = circuit.assign_parameters({"u_theta_0": theta}).to_matrix()[:, 0]
        return float(np.vdot(state, observable.to_matrix() @ state).real)

    step = 1e-6
    expected = (expectation(0.31 + step) - expectation(0.31 - step)) / (2 * step)
    assert abs(expected) > 0.1
    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: UCCAnsatz(()), ValueError, "must not be empty"),
        (lambda: UCCAnsatz("bad"), TypeError, "sequence"),
        (
            lambda: UCCAnsatz((FermionicExcitation(4, (0,), (2,)), object())),
            TypeError,
            "FermionicExcitation",
        ),
        (
            lambda: UCCAnsatz(
                (
                    FermionicExcitation(4, (0,), (2,)),
                    FermionicExcitation(6, (0,), (2,)),
                )
            ),
            ValueError,
            "same number",
        ),
        (
            lambda: UCCAnsatz((FermionicExcitation(4, (0,), (2,)),), mapper=object()),
            TypeError,
            "FermionToQubitMapper",
        ),
        (
            lambda: UCCAnsatz(
                (FermionicExcitation(4, (0,), (2,)),), initial_state=object()
            ),
            TypeError,
            "StatePreparation",
        ),
        (
            lambda: UCCAnsatz(
                (FermionicExcitation(4, (0,), (2,)),),
                initial_state=BasisStateInitialization(3, 0),
            ),
            ValueError,
            "same width",
        ),
    ],
)
def test_invalid_primary_inputs(factory, error, message):
    with pytest.raises(error, match=message):
        factory()


class _OutputMapper:
    def __init__(self, width: object, output: object) -> None:
        self.width = width
        self.output = output

    def num_qubits(self, num_modes: int) -> int:
        return self.width  # type: ignore[return-value]

    def map_operator(self, operator: FermionicOp) -> Hamiltonian:
        return self.output  # type: ignore[return-value]


def _hamiltonian(width: int, *terms: tuple[str, complex]) -> Hamiltonian:
    result = Hamiltonian(width)
    for label, coefficient in terms:
        result.add_term(PauliString.from_str(label), coefficient)
    return result


@pytest.mark.parametrize(
    ("mapper", "error", "message"),
    [
        (
            _OutputMapper("4", _hamiltonian(4, ("XIII", 1.0))),
            TypeError,
            "return an int",
        ),
        (_OutputMapper(0, _hamiltonian(1, ("X", 1.0))), ValueError, "positive width"),
        (_OutputMapper(4, object()), TypeError, "return a Hamiltonian"),
        (
            _OutputMapper(4, _hamiltonian(3, ("XII", 1.0))),
            ValueError,
            "match mapper num_qubits",
        ),
        (_OutputMapper(4, Hamiltonian(4)), ValueError, "must not be empty"),
        (
            _OutputMapper(4, _hamiltonian(4, ("XIII", 1.0j))),
            ValueError,
            "not Hermitian",
        ),
        (
            _OutputMapper(4, _hamiltonian(4, ("XIII", 1.0), ("ZIII", 1.0))),
            ValueError,
            "must commute",
        ),
    ],
)
def test_invalid_mapper_results_are_rejected(mapper, error, message):
    with pytest.raises(error, match=message):
        UCCAnsatz((FermionicExcitation(4, (0,), (2,)),), mapper=mapper)


def test_invalid_initial_state_contracts_are_rejected():
    excitation = (FermionicExcitation(4, (0,), (2,)),)
    with pytest.raises(ValueError, match="zero logical parameters"):
        UCCAnsatz(
            excitation,
            initial_state=_FixedCircuitPreparation(Circuit(4), width=4, parameters=1),
        )

    symbolic = Circuit(4)
    symbolic.ry(0, Parameter("active"))
    with pytest.raises(ValueError, match="active symbols"):
        UCCAnsatz(
            excitation,
            initial_state=_FixedCircuitPreparation(symbolic, width=4),
        )
    with pytest.raises(ValueError, match="inconsistent width"):
        UCCAnsatz(
            excitation,
            initial_state=_FixedCircuitPreparation(Circuit(3), width=4),
        )
    with pytest.raises(TypeError, match="build a Circuit"):
        UCCAnsatz(
            excitation,
            initial_state=_FixedCircuitPreparation(object(), width=4),
        )


@pytest.mark.parametrize("operation", ["measure", "reset", "nested"])
def test_noninvertible_initial_state_circuits_are_rejected(operation):
    source = Circuit(4)
    if operation == "measure":
        source.measure(0)
    elif operation == "reset":
        source.reset(0)
    else:
        inner = Circuit(1)
        inner.measure(0)
        source.append_circuit_gate(inner.to_gate("hidden_measure"), [0])

    with pytest.raises(ValueError, match="statically invertible"):
        UCCAnsatz(
            (FermionicExcitation(4, (0,), (2,)),),
            initial_state=_FixedCircuitPreparation(source, width=4),
        )


def test_prefix_validation_and_build_independence():
    component = UCCAnsatz((FermionicExcitation(4, (0,), (2,)),))
    with pytest.raises(TypeError, match="prefix"):
        component.build_circuit(1)  # type: ignore[arg-type]
    first = component.build_circuit("u")
    second = component.build_circuit("u")
    first.x(0)
    assert first != second
