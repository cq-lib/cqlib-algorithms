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

"""Tests for the Grover operator composite component."""

from __future__ import annotations

import math
from dataclasses import is_dataclass

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import ClassicalExpr
from cqlib.circuit.gates import UnitaryGate

from cqlib_algorithms.operators.composite import GroverOperator
from cqlib_algorithms.operators.oracle import (
    BooleanFunction,
    GroverDiffusion,
    PhaseOracle,
)


def _phase_oracle_matrix(num_qubits, marked_states):
    dimension = 1 << num_qubits
    diagonal = np.ones(dimension, dtype=np.complex128)
    for state in marked_states:
        diagonal[state] = -1.0
    return np.diag(diagonal)


def _uniform_diffusion_matrix(num_qubits):
    dimension = 1 << num_qubits
    uniform = np.ones(dimension, dtype=np.complex128) / math.sqrt(dimension)
    return 2.0 * np.outer(uniform, uniform.conj()) - np.eye(
        dimension, dtype=np.complex128
    )


def _partial_diffusion_matrix(num_qubits, reflection_qubits):
    """Reference ``A S_0 A†`` with ``A`` placing Hadamards on the subset."""
    dimension = 1 << num_qubits
    subset = set(reflection_qubits)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    identity = np.eye(2, dtype=np.complex128)
    factors = [hadamard if qubit in subset else identity for qubit in range(num_qubits)]
    preparation = factors[0]
    for factor in factors[1:]:
        preparation = np.kron(factor, preparation)

    zero_indices = [
        index
        for index in range(dimension)
        if all(not (index >> qubit) & 1 for qubit in subset)
    ]
    projector = np.zeros((dimension, dimension), dtype=np.complex128)
    projector[zero_indices, zero_indices] = 1.0
    zero_reflection = 2.0 * projector - np.eye(dimension, dtype=np.complex128)
    return preparation @ zero_reflection @ preparation.conj().T


def _state_reflection_matrix(state_vector):
    dimension = state_vector.shape[0]
    return 2.0 * np.outer(state_vector, state_vector.conj()) - np.eye(
        dimension, dtype=np.complex128
    )


def _zero_subspace_reflection_matrix(num_qubits, reflection_qubits):
    """Return ``2 P_(0,R) - I`` from its independent basis definition."""
    dimension = 1 << num_qubits
    diagonal = np.full(dimension, -1.0, dtype=np.complex128)
    for index in range(dimension):
        if all(((index >> qubit) & 1) == 0 for qubit in reflection_qubits):
            diagonal[index] = 1.0
    return np.diag(diagonal)


def _prepared_subspace_reflection_matrix(preparation, reflection_qubits):
    num_qubits = preparation.shape[0].bit_length() - 1
    zero_subspace = _zero_subspace_reflection_matrix(num_qubits, reflection_qubits)
    return preparation @ zero_subspace @ preparation.conj().T


def _truth_table(num_qubits, marked_states):
    values = [False] * (1 << num_qubits)
    for state in marked_states:
        values[state] = True
    return values


def _phase_oracle_circuit(num_qubits, marked_states):
    function = BooleanFunction.from_truth_table(_truth_table(num_qubits, marked_states))
    return PhaseOracle(function).build_circuit()


def test_partial_diffusion_reference_is_consistent_with_uniform_reference():
    for num_qubits in (1, 2, 3):
        np.testing.assert_allclose(
            _partial_diffusion_matrix(num_qubits, tuple(range(num_qubits))),
            _uniform_diffusion_matrix(num_qubits),
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("num_qubits", "marked_states"),
    [
        (1, (0,)),
        (2, (1,)),
        (2, (0, 3)),
        (3, (2, 5)),
        (4, (1, 7, 12)),
    ],
)
def test_grover_operator_matches_independent_matrix_reference(
    num_qubits, marked_states
):
    oracle = _phase_oracle_circuit(num_qubits, marked_states)
    operator = GroverOperator(oracle)

    expected = _uniform_diffusion_matrix(num_qubits) @ _phase_oracle_matrix(
        num_qubits, marked_states
    )
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_default_diffusion_composes_oracle_and_grover_diffusion(num_qubits):
    marked_states = tuple(range(0, 1 << num_qubits, 3))
    oracle = _phase_oracle_circuit(num_qubits, marked_states)
    operator = GroverOperator(oracle)

    expected = GroverDiffusion(num_qubits).build_circuit().to_matrix() @ (
        _phase_oracle_matrix(num_qubits, marked_states)
    )
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_amplitude_amplification_matches_analytic_success_probability():
    num_qubits = 4
    marked_states = (1, 7, 12)
    operator = GroverOperator(_phase_oracle_circuit(num_qubits, marked_states))
    matrix = operator.build_circuit().to_matrix()

    dimension = 1 << num_qubits
    initial = np.ones(dimension, dtype=np.complex128) / math.sqrt(dimension)
    theta = math.asin(math.sqrt(len(marked_states) / dimension))

    state = initial.copy()
    for iterations in range(3):
        if iterations > 0:
            state = matrix @ state
        expected = math.sin((2 * iterations + 1) * theta) ** 2
        measured = sum(abs(state[marked]) ** 2 for marked in marked_states)
        assert math.isclose(measured, expected, rel_tol=0.0, abs_tol=1e-12)


def test_custom_state_preparation_reflects_about_prepared_state():
    # A = H on qubit 0 and X on qubit 1 prepares |+>|1>, whose amplitudes
    # sit at basis indices 2 and 3 under the little-endian convention.
    preparation = Circuit(2)
    preparation.h(0)
    preparation.x(1)
    marked_states = (0, 3)
    operator = GroverOperator(
        _phase_oracle_circuit(2, marked_states),
        state_preparation=preparation,
    )

    state_vector = np.zeros(4, dtype=np.complex128)
    state_vector[2] = 1.0 / math.sqrt(2.0)
    state_vector[3] = 1.0 / math.sqrt(2.0)
    expected = _state_reflection_matrix(state_vector) @ _phase_oracle_matrix(
        2, marked_states
    )
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_custom_state_preparation_without_subset_matches_full_reflection():
    preparation = Circuit(2)
    preparation.h(0)
    preparation.h(1)
    marked_states = (1,)
    operator = GroverOperator(
        _phase_oracle_circuit(2, marked_states),
        state_preparation=preparation,
    )

    expected = _uniform_diffusion_matrix(2) @ _phase_oracle_matrix(2, marked_states)
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_entangled_custom_preparation_partial_subset_matches_projector_formula():
    preparation = Circuit(3)
    preparation.h(0)
    preparation.cx(0, 1)
    preparation.ry(2, 0.417)
    preparation.cz(1, 2)
    reflection_qubits = (0, 2)
    marked_states = (1, 6)

    operator = GroverOperator(
        _phase_oracle_circuit(3, marked_states),
        state_preparation=preparation,
        reflection_qubits=reflection_qubits,
    )

    diffusion = _prepared_subspace_reflection_matrix(
        preparation.to_matrix(), reflection_qubits
    )
    expected = diffusion @ _phase_oracle_matrix(3, marked_states)
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(), expected, rtol=0.0, atol=1e-12
    )


def test_custom_partial_reflection_has_the_expected_positive_subspace_dimension():
    preparation = Circuit(3)
    preparation.h(0)
    preparation.cx(0, 1)
    preparation.ry(2, -0.283)
    reflection_qubits = (1,)

    diffusion = _prepared_subspace_reflection_matrix(
        preparation.to_matrix(), reflection_qubits
    )
    eigenvalues = np.linalg.eigvalsh(diffusion)

    assert np.count_nonzero(np.isclose(eigenvalues, 1.0, atol=1e-12)) == 4
    assert np.count_nonzero(np.isclose(eigenvalues, -1.0, atol=1e-12)) == 4
    single_state = _state_reflection_matrix(preparation.to_matrix()[:, 0])
    assert not np.allclose(diffusion, single_state, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("num_qubits", "reflection_qubits"),
    [
        (2, (0,)),
        (2, (1,)),
        (3, (0, 2)),
        (3, (1, 2)),
    ],
)
def test_partial_reflection_qubits_match_independent_reference(
    num_qubits, reflection_qubits
):
    marked_states = (1,)
    operator = GroverOperator(
        _phase_oracle_circuit(num_qubits, marked_states),
        reflection_qubits=reflection_qubits,
    )

    expected = _partial_diffusion_matrix(
        num_qubits, reflection_qubits
    ) @ _phase_oracle_matrix(num_qubits, marked_states)
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_full_reflection_qubits_subset_equals_default():
    marked_states = (2,)
    oracle = _phase_oracle_circuit(3, marked_states)
    default = GroverOperator(oracle)
    explicit = GroverOperator(oracle, reflection_qubits=(0, 1, 2))

    np.testing.assert_allclose(
        explicit.build_circuit().to_matrix(),
        default.build_circuit().to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )


def test_reflection_qubits_are_sorted_and_frozen():
    operator = GroverOperator(
        _phase_oracle_circuit(3, (0,)),
        reflection_qubits=(2, 0, 1),
    )

    assert operator.reflection_qubits == (0, 1, 2)


def test_reflection_qubit_generator_is_consumed_exactly_once():
    class SingleUseIndices:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("reflection_qubits was consumed more than once")
            yield 2
            yield 0

    indices = SingleUseIndices()
    operator = GroverOperator(_phase_oracle_circuit(3, (1,)), reflection_qubits=indices)

    assert indices.iterations == 1
    assert operator.reflection_qubits == (0, 2)


def test_grover_operator_is_unitary():
    operator = GroverOperator(_phase_oracle_circuit(3, (1, 5)))
    matrix = operator.build_circuit().to_matrix()
    identity = np.eye(1 << 3)

    np.testing.assert_allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-12)


def test_oracle_with_non_contiguous_qubit_ids_is_mapped_positionally():
    oracle = _phase_oracle_circuit(2, (2,))
    remapped = Circuit([5, 9])
    remapped.compose(oracle, [Qubit(5), Qubit(9)])

    operator = GroverOperator(remapped)

    expected = _uniform_diffusion_matrix(2) @ _phase_oracle_matrix(2, (2,))
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_oracle_and_preparation_with_different_qubit_ids_map_by_position():
    compact_oracle = _phase_oracle_circuit(2, (2,))
    oracle = Circuit([9, 4])
    oracle.compose(compact_oracle, [Qubit(9), Qubit(4)])

    preparation = Circuit([20, 7])
    preparation.ry(20, 0.351)
    preparation.cx(20, 7)
    preparation_matrix = preparation.to_matrix([20, 7])

    operator = GroverOperator(oracle, state_preparation=preparation)
    expected = _prepared_subspace_reflection_matrix(
        preparation_matrix, (0, 1)
    ) @ _phase_oracle_matrix(2, (2,))

    assert [qubit.index for qubit in operator.oracle.qubits] == [0, 1]
    assert operator.state_preparation is not None
    assert [qubit.index for qubit in operator.state_preparation.qubits] == [0, 1]
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(), expected, rtol=0.0, atol=1e-12
    )


def test_oracle_is_defensively_copied():
    oracle = _phase_oracle_circuit(2, (1,))
    operator = GroverOperator(oracle)
    oracle.x(0)
    oracle.set_global_phase(math.pi)

    expected = _uniform_diffusion_matrix(2) @ _phase_oracle_matrix(2, (1,))
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_oracle_accessor_returns_an_independent_normalized_copy():
    compact = _phase_oracle_circuit(2, (1,))
    source = Circuit([7, 12])
    source.compose(compact, [Qubit(7), Qubit(12)])
    operator = GroverOperator(source)

    exposed = operator.oracle
    assert [qubit.index for qubit in exposed.qubits] == [0, 1]
    exposed.x(0)

    expected = _uniform_diffusion_matrix(2) @ _phase_oracle_matrix(2, (1,))
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(), expected, rtol=0.0, atol=1e-12
    )


def test_symbolic_parameters_and_global_phases_preserve_exact_matrix_semantics():
    phi = Parameter("phi")
    theta = Parameter("theta")

    oracle = Circuit(2)
    oracle.rz(0, phi)
    oracle.cz(0, 1)
    oracle.set_global_phase(0.219)

    preparation = Circuit(2)
    preparation.ry(0, theta)
    preparation.cx(0, 1)
    preparation.set_global_phase(-0.713)

    built = GroverOperator(oracle, state_preparation=preparation).build_circuit()
    bindings = {"phi": -0.401, "theta": 0.638}
    oracle_matrix = oracle.assign_parameters(bindings).to_matrix()
    preparation_matrix = preparation.assign_parameters(bindings).to_matrix()
    expected = (
        _prepared_subspace_reflection_matrix(preparation_matrix, (0, 1)) @ oracle_matrix
    )

    assert set(built.used_symbols) == {"phi", "theta"}
    np.testing.assert_allclose(
        built.assign_parameters(bindings).to_matrix(), expected, rtol=0.0, atol=1e-12
    )


def test_barriers_are_allowed_and_preserved_in_oracle_and_preparation():
    oracle = _phase_oracle_circuit(1, (0,))
    oracle.barrier([0])
    preparation = Circuit(1)
    preparation.h(0)
    preparation.barrier([0])

    built = GroverOperator(oracle, state_preparation=preparation).build_circuit()

    assert [operation.name for operation in built.operations].count("barrier") == 3
    expected = _state_reflection_matrix(
        np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    ) @ _phase_oracle_matrix(1, (0,))
    np.testing.assert_allclose(built.to_matrix(), expected, rtol=0.0, atol=1e-12)


def test_build_returns_fresh_independent_circuits():
    operator = GroverOperator(_phase_oracle_circuit(2, (1,)))
    first = operator.build_circuit()
    second = operator.build_circuit()

    assert first is not second
    assert first == second
    second_size = len(second.operations)
    first.x(0)
    assert len(second.operations) == second_size


def test_grover_operator_is_a_slotted_component_with_read_only_config():
    operator = GroverOperator(_phase_oracle_circuit(2, (1,)))

    assert not is_dataclass(operator)
    assert not hasattr(operator, "__dict__")
    assert operator.num_qubits == 2
    assert operator.reflection_qubits == (0, 1)
    assert operator.state_preparation is None
    with pytest.raises(AttributeError):
        operator.num_qubits = 3  # type: ignore[misc]


def test_state_preparation_accessor_returns_an_independent_copy():
    preparation = Circuit(1)
    preparation.h(0)
    operator = GroverOperator(
        _phase_oracle_circuit(1, (0,)), state_preparation=preparation
    )

    accessed = operator.state_preparation
    assert accessed is not None
    accessed.x(0)
    np.testing.assert_allclose(
        operator.build_circuit().to_matrix(),
        _state_reflection_matrix(
            np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
        )
        @ _phase_oracle_matrix(1, (0,)),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("oracle", [None, "circuit", 2, 1.5])
def test_grover_operator_rejects_non_circuit_oracles(oracle):
    with pytest.raises(TypeError, match="oracle must be a Circuit"):
        GroverOperator(oracle)  # type: ignore[arg-type]


def test_grover_operator_rejects_empty_oracle():
    with pytest.raises(ValueError, match="at least one qubit"):
        GroverOperator(Circuit(0))


def test_grover_operator_rejects_non_invertible_oracle():
    oracle = Circuit(1)
    oracle.reset(0)

    with pytest.raises(ValueError, match="valid invertible operations"):
        GroverOperator(oracle)


def _invalid_circuit(kind):
    num_qubits = 3 if kind == "unitary" else 1
    circuit = Circuit(num_qubits)
    if kind == "measure":
        circuit.measure(0)
    elif kind == "reset":
        circuit.reset(0)
    elif kind == "delay":
        circuit.delay(0, 1.0)
    elif kind == "control":
        circuit.if_(ClassicalExpr.bool_literal(True), lambda body: body.x(0))
    elif kind == "nested":
        nested = Circuit(1)
        nested.measure(0)
        circuit.append_circuit_gate(nested.to_gate("non_unitary"), [0])
    else:
        gate = UnitaryGate("identity-3q", 3).with_matrix(np.eye(8, dtype=np.complex128))
        circuit.append_unitary_gate(gate, [0, 1, 2])
    return circuit


@pytest.mark.parametrize(
    "kind", ["measure", "reset", "delay", "control", "nested", "unitary"]
)
def test_grover_operator_rejects_dynamic_or_noninvertible_oracle_paths(kind):
    with pytest.raises(ValueError, match="oracle must contain only valid invertible"):
        GroverOperator(_invalid_circuit(kind))


@pytest.mark.parametrize("state_preparation", ["circuit", 2])
def test_grover_operator_rejects_non_circuit_state_preparation(
    state_preparation,
):
    oracle = _phase_oracle_circuit(1, (0,))
    with pytest.raises(TypeError, match="state_preparation must be a Circuit"):
        GroverOperator(oracle, state_preparation=state_preparation)  # type: ignore[arg-type]


def test_grover_operator_rejects_mismatched_state_preparation_size():
    preparation = Circuit(3)
    preparation.h(0)

    with pytest.raises(ValueError, match="same number of qubits as oracle"):
        GroverOperator(_phase_oracle_circuit(2, (1,)), state_preparation=preparation)


def test_grover_operator_rejects_non_invertible_state_preparation():
    preparation = Circuit(1)
    preparation.reset(0)

    with pytest.raises(ValueError, match="valid invertible operations"):
        GroverOperator(_phase_oracle_circuit(1, (0,)), state_preparation=preparation)


@pytest.mark.parametrize(
    "kind", ["measure", "reset", "delay", "control", "nested", "unitary"]
)
def test_grover_operator_rejects_dynamic_or_noninvertible_preparation_paths(kind):
    preparation = _invalid_circuit(kind)
    oracle = _phase_oracle_circuit(int(preparation.num_qubits), (0,))

    with pytest.raises(
        ValueError, match="state_preparation must contain only valid invertible"
    ):
        GroverOperator(oracle, state_preparation=preparation)


@pytest.mark.parametrize("reflection_qubits", [1, 2.5, True])
def test_grover_operator_rejects_non_iterable_reflection_qubits(
    reflection_qubits,
):
    with pytest.raises(
        TypeError, match="reflection_qubits must be an iterable of integers"
    ):
        GroverOperator(
            _phase_oracle_circuit(1, (0,)),
            reflection_qubits=reflection_qubits,  # type: ignore[arg-type]
        )


def test_grover_operator_rejects_string_reflection_qubits():
    with pytest.raises(
        TypeError, match="reflection_qubits must be an iterable of integers"
    ):
        GroverOperator(_phase_oracle_circuit(2, (0,)), reflection_qubits="01")


@pytest.mark.parametrize("reflection_qubits", [[], ()])
def test_grover_operator_rejects_empty_reflection_qubits(reflection_qubits):
    with pytest.raises(ValueError, match="must not be empty"):
        GroverOperator(
            _phase_oracle_circuit(2, (0,)), reflection_qubits=reflection_qubits
        )


@pytest.mark.parametrize("reflection_qubits", [[True], [0, 1.5], ["0"]])
def test_grover_operator_rejects_non_integer_reflection_qubits(
    reflection_qubits,
):
    with pytest.raises(TypeError, match="must contain only integers"):
        GroverOperator(
            _phase_oracle_circuit(2, (0,)), reflection_qubits=reflection_qubits
        )


@pytest.mark.parametrize("reflection_qubits", [[-1], [0, 2], [3]])
def test_grover_operator_rejects_out_of_range_reflection_qubits(
    reflection_qubits,
):
    with pytest.raises(ValueError, match="valid search-qubit indices"):
        GroverOperator(
            _phase_oracle_circuit(2, (0,)), reflection_qubits=reflection_qubits
        )


def test_grover_operator_rejects_duplicate_reflection_qubits():
    with pytest.raises(ValueError, match="must not contain duplicates"):
        GroverOperator(_phase_oracle_circuit(2, (0,)), reflection_qubits=[0, 1, 1])
