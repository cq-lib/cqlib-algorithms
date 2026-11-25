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

# Modified to use the shared fidelity quantum-kernel operator.

"""Analytic fidelity references, pair ordering, and resource preflight."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.circuit import CircuitError
from cqlib.circuit.gates import UnitaryGate
from cqlib.device import ExecutionResult

from cqlib_algorithms.operators.quantum_kernel import FidelityQuantumKernel
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import UserCircuitPreparation
from cqlib_algorithms.protocols import Sampler


def ry_kernel(**kwargs):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    return FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)


def analytic_ry(x, y):
    return np.cos((np.asarray(x)[:, None, 0] - np.asarray(y)[None, :, 0]) / 2) ** 2


@pytest.mark.reference
def test_ry_dense_grid_and_rectangular_shuffled_reference():
    kernel = ry_kernel()
    x = np.array([[0], [math.pi / 2], [math.pi], [0.13], [-0.92], [2.41]])
    y = np.array([[2.41], [0.37], [0], [-2.2]])
    np.testing.assert_allclose(
        kernel.evaluate(x), analytic_ry(x, x), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        kernel.evaluate(x, y), analytic_ry(x, y), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(kernel.evaluate(x), kernel.evaluate(x))
    assert np.array_equal(kernel.evaluate(x), kernel.evaluate(x).T)


@pytest.mark.reference
def test_product_map_explicit_nonalphabetic_order_repetition_and_fixed_binding():
    circuit = Circuit([7, 2])
    circuit.ry(Qubit(7), Parameter("z"))
    circuit.ry(Qubit(7), Parameter("z"))
    circuit.ry(Qubit(2), Parameter("a"))
    circuit.rz(Qubit(2), Parameter("fixed"))
    circuit.set_global_phase(0.73)
    fixed = {"fixed": 0.37}
    kernel = FidelityQuantumKernel(
        UserCircuitPreparation(circuit), ["z", "a"], parameter_bindings=fixed
    )
    x = np.array([[0.1, 0.7], [0.3, -0.2], [1.3, 0.11]])
    expected = np.cos(x[:, None, 0] - x[None, :, 0]) ** 2
    expected *= np.cos((x[:, None, 1] - x[None, :, 1]) / 2) ** 2
    fixed["fixed"] = 4.0
    circuit.x(Qubit(7))
    kernel.feature_map.x(0)
    np.testing.assert_allclose(kernel.evaluate(x), expected, atol=1e-12, rtol=0)
    assert kernel.input_parameters == ("z", "a")
    assert kernel.parameter_bindings == {"fixed": 0.37}
    with pytest.raises(TypeError):
        kernel.parameter_bindings["fixed"] = 4.0


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("fixed_scale", [False, True])
def test_nonzero_feature_domain_matches_reciprocal_analytic_kernel(
    sampled, fixed_scale
):
    parameter = Parameter("x")
    scale = 2.0 if fixed_scale else 1.0
    circuit = Circuit(1)
    circuit.ry(0, Parameter("scale") / parameter if fixed_scale else 1 / parameter)
    data = np.array([[1.0], [2.0], [1.0]])
    angles = scale / data[:, 0]
    expected = np.cos((angles[:, None] - angles[None, :]) / 2) ** 2
    shots = 1000
    zero = round(shots * expected[0, 1])
    sampler = PairSampler([zero, zero])
    kwargs = {"parameter_bindings": {"scale": scale}} if fixed_scale else {}
    if sampled:
        kwargs.update(sampler=sampler, shots=shots)
    kernel = FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)
    matrix = kernel.evaluate(data)
    if sampled:
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=0.5 / shots)
        for request, (left, right) in zip(sampler.calls[0], [(0, 1), (1, 2)]):
            difference = angles[right] - angles[left]
            expected_unitary = np.array(
                [
                    [math.cos(difference / 2), -math.sin(difference / 2)],
                    [math.sin(difference / 2), math.cos(difference / 2)],
                ]
            )
            np.testing.assert_allclose(
                request.circuit.to_matrix(), expected_unitary, rtol=0, atol=1e-12
            )
    else:
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("sampled", [False, True])
def test_nonzero_domain_equal_and_empty_rows_do_not_execute(sampled, monkeypatch):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    circuit = Circuit(1)
    circuit.ry(0, 1 / Parameter("x"))
    sampler = PairSampler([])
    kwargs = {"sampler": sampler, "shots": 10} if sampled else {}
    kernel = FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)
    monkeypatch.setattr(
        module.Statevector, "from_circuit", lambda *a: pytest.fail("no execution")
    )
    np.testing.assert_array_equal(kernel.evaluate([[1.0], [1.0]], [[1.0]]), [[1], [1]])
    monkeypatch.setattr(
        FidelityQuantumKernel,
        "_bound",
        lambda *a: pytest.fail("empty evaluation bound"),
    )
    assert kernel.evaluate(np.empty((0, 1))).shape == (0, 0)
    assert kernel.evaluate(np.empty((0, 1)), [[0.0]]).shape == (0, 1)
    assert kernel.evaluate([[0.0]], np.empty((0, 1))).shape == (1, 0)
    assert sampler.calls == []


def test_constructor_checks_fixed_operations_without_binding_input(monkeypatch):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    circuit = Circuit(1)
    circuit.ry(0, 1 / Parameter("x"))
    circuit.rz(0, Parameter("fixed"))
    checked = []
    original = module.validate_bound_static_unitary_circuit

    def record(candidate, name):
        checked.append(candidate)
        original(candidate, name)

    monkeypatch.setattr(module, "validate_bound_static_unitary_circuit", record)
    FidelityQuantumKernel(
        UserCircuitPreparation(circuit), ["x"], parameter_bindings={"fixed": 0.3}
    )
    assert len(checked) == 1
    assert not checked[0].used_symbols
    expected = Circuit(1)
    expected.rz(0, 0.3)
    np.testing.assert_allclose(checked[0].to_matrix(), expected.to_matrix(), atol=1e-12)


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("kind,value", [("singularity", 0.0), ("overflow", 1e308)])
def test_invalid_fixed_only_expression_rejected_by_constructor(sampled, kind, value):
    parameter = Parameter("fixed")
    circuit = Circuit(1)
    circuit.ry(0, 1 / Parameter("x"))
    circuit.rz(0, 1 / parameter if kind == "singularity" else parameter * parameter)
    kwargs = {"sampler": PairSampler([]), "shots": 10} if sampled else {}
    with pytest.raises((CircuitError, ValueError)):
        FidelityQuantumKernel(
            UserCircuitPreparation(circuit),
            ["x"],
            parameter_bindings={"fixed": value},
            **kwargs,
        )


@pytest.mark.parametrize("sampled", [False, True])
def test_unsupported_numeric_unitary_inverse_is_still_rejected(sampled):
    circuit = Circuit(3)
    circuit.ry(0, 1 / Parameter("x"))
    gate = UnitaryGate("numeric", 3).with_matrix(np.eye(8))
    circuit.append_unitary_gate(gate, [0, 1, 2])
    kwargs = {"sampler": PairSampler([]), "shots": 10} if sampled else {}
    with pytest.raises(ValueError, match="invertible"):
        FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)


class PairSampler(Sampler):
    def __init__(self, zeros):
        self.zeros = list(zeros)
        self.calls = []

    def run(self, requests):
        self.calls.append(tuple(requests))
        assert len(requests) == len(self.zeros)
        return [
            ExecutionResult.from_counts(
                "pair",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                {"0": zero, "1": request.shots - zero},
            )
            for request, zero in zip(requests, self.zeros)
        ]


def test_sampled_upper_triangle_pair_bindings_duplicates_and_delegate(monkeypatch):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    original = module.ComputeUncomputeFidelity.run
    batches = []

    def record(self, requests):
        batches.append(requests)
        return original(self, requests)

    monkeypatch.setattr(module.ComputeUncomputeFidelity, "run", record)
    sampler = PairSampler([2, 3, 7, 5, 9])
    kernel = ry_kernel(sampler=sampler, shots=10)
    x = [[0], [0.5], [0], [1.2]]
    matrix = kernel.evaluate(x)
    assert len(batches) == len(sampler.calls) == 1
    pairs = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]
    expected = np.ones((4, 4))
    for request, (i, j), value in zip(
        sampler.calls[0], pairs, [0.2, 0.3, 0.7, 0.5, 0.9]
    ):
        expected[i, j] = expected[j, i] = value
        assert request.shots == 10
        # The submitted unitary is RY(-x_i) RY(x_j), not an unrelated pair.
        angle = x[j][0] - x[i][0]
        expected_circuit = np.array(
            [
                [math.cos(angle / 2), -math.sin(angle / 2)],
                [math.sin(angle / 2), math.cos(angle / 2)],
            ]
        )
        np.testing.assert_allclose(
            request.circuit.to_matrix(), expected_circuit, atol=1e-12
        )
    np.testing.assert_array_equal(matrix, expected)


def test_sampled_rectangular_order_and_identical_row_shortcut():
    sampler = PairSampler([1, 4, 9])
    kernel = ry_kernel(sampler=sampler, shots=10)
    np.testing.assert_array_equal(
        kernel.evaluate([[0], [1]], [[1], [2]]), [[0.1, 0.4], [1, 0.9]]
    )
    assert len(sampler.calls[0]) == 3


@pytest.mark.integration
def test_real_sampler_same_and_orthogonal_states_are_deterministic():
    kernel = ry_kernel(sampler=StatevectorSampler(), shots=17)
    np.testing.assert_array_equal(
        kernel.evaluate([[0], [math.pi], [0]]), [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    )


@pytest.mark.parametrize(
    "x",
    [
        [],
        [0, 1],
        [[[1]]],
        [[1, 2]],
        [[True]],
        [[1j]],
        [["1"]],
        np.array([[1]], dtype=object),
        [[math.inf]],
        [[math.nan]],
        [[1], [1, 2]],
    ],
)
def test_invalid_feature_arrays(x):
    with pytest.raises((TypeError, ValueError)):
        ry_kernel().evaluate(x)


@pytest.mark.parametrize("which", ["X", "Y"])
def test_both_inputs_are_validated(which):
    with pytest.raises(ValueError, match="finite"):
        ry_kernel().evaluate(
            [[math.nan]] if which == "X" else [[0]],
            [[math.nan]] if which == "Y" else [[0]],
        )


def test_empty_shapes_no_quantum_execution(monkeypatch):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    monkeypatch.setattr(
        module.Statevector, "from_circuit", lambda *a: pytest.fail("no execution")
    )
    kernel = ry_kernel()
    assert kernel.evaluate(np.empty((0, 1))).shape == (0, 0)
    assert kernel.evaluate(np.empty((0, 1)), [[0], [1]]).shape == (0, 2)
    assert kernel.evaluate([[0], [1]], np.empty((0, 1))).shape == (2, 0)


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize(
    "kind,value", [("singularity", 1.0), ("overflow", 1e308), ("reciprocal", 0.0)]
)
@pytest.mark.parametrize("shape", ["single", "duplicates", "cross"])
def test_equal_rows_reject_invalid_bound_feature_states(
    sampled, kind, value, shape, monkeypatch
):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    parameter = Parameter("x")
    circuit = Circuit(1)
    angle = (
        1 / parameter
        if kind == "reciprocal"
        else parameter / (parameter - 1)
        if kind == "singularity"
        else parameter * parameter
    )
    circuit.ry(0, angle)
    sampler = PairSampler([])
    kwargs = {"sampler": sampler, "shots": 10} if sampled else {}
    kernel = FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)
    monkeypatch.setattr(
        module.Statevector,
        "from_circuit",
        lambda *a: pytest.fail("invalid state executed"),
    )
    left = [[value], [value]] if shape == "duplicates" else [[value]]
    right = [[value]] if shape == "cross" else None
    with pytest.raises((CircuitError, ValueError)):
        kernel.evaluate(left, right)
    assert sampler.calls == []


@pytest.mark.parametrize("sampled", [False, True])
def test_equal_rows_validate_one_circuit_per_distinct_row_without_execution(
    sampled, monkeypatch
):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    sampler = PairSampler([])
    kwargs = {"sampler": sampler, "shots": 10} if sampled else {}
    kernel = ry_kernel(**kwargs)
    original = FidelityQuantumKernel._bound
    bindings = []

    def record(self, row):
        bindings.append(row)
        return original(self, row)

    monkeypatch.setattr(FidelityQuantumKernel, "_bound", record)
    monkeypatch.setattr(
        module.Statevector,
        "from_circuit",
        lambda *a: pytest.fail("equal rows executed"),
    )
    np.testing.assert_array_equal(kernel.evaluate([[0.3], [0.3]], [[0.3]]), [[1], [1]])
    assert bindings == [(0.3,)]
    assert sampler.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_parameters": []},
        {"input_parameters": "x"},
        {"input_parameters": ["x", "x"]},
        {"input_parameters": [1]},
        {"input_parameters": ["unknown"]},
        {"parameter_bindings": {"x": 0.2}},
        {"parameter_bindings": {"extra": 1}},
        {"parameter_bindings": {"fixed": math.nan}},
        {"parameter_bindings": {"fixed": True}},
        {"parameter_bindings": []},
        {"parameter_bindings": {1: 0.2}},
        {"sampler": StatevectorSampler()},
        {"shots": 10},
        {"sampler": object(), "shots": 10},
        {"sampler": StatevectorSampler(), "shots": True},
        {"sampler": StatevectorSampler(), "shots": 0},
        {"max_statevector_qubits": 0},
        {"max_statevector_bytes": True},
    ],
)
def test_invalid_kernel_configuration(kwargs):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    arguments = {"input_parameters": ["x"], **kwargs}
    with pytest.raises((TypeError, ValueError)):
        FidelityQuantumKernel(UserCircuitPreparation(circuit), **arguments)


@pytest.mark.parametrize("kind", ["measure", "reset", "delay", "nested"])
def test_nonunitary_envelope_is_rejected_in_both_modes(kind):
    circuit = Circuit(1)
    circuit.ry(0, Parameter("x"))
    if kind == "measure":
        circuit.measure(0)
    elif kind == "reset":
        circuit.reset(0)
    elif kind == "delay":
        circuit.delay(0, 1.0)
    else:
        inner = Circuit(1)
        inner.measure(0)
        circuit.compose(inner, circuit.qubits)
    for kwargs in ({}, {"sampler": StatevectorSampler(), "shots": 10}):
        with pytest.raises(ValueError):
            FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)


@pytest.mark.parametrize("budget", ["qubits", "bytes"])
def test_exact_limits_reject_before_core_simulation(budget, monkeypatch):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    calls = []
    monkeypatch.setattr(module.Statevector, "from_circuit", lambda *a: calls.append(a))
    circuit = Circuit(2)
    circuit.ry(0, Parameter("x"))
    kwargs = (
        {"max_statevector_qubits": 1}
        if budget == "qubits"
        else {"max_statevector_bytes": 127}
    )
    kernel = FidelityQuantumKernel(UserCircuitPreparation(circuit), ["x"], **kwargs)
    with pytest.raises(ValueError, match="max_statevector"):
        kernel.evaluate([[0]], [[1]])
    assert calls == []


def test_exact_byte_budget_counts_both_sets_and_deduplicates_rows():
    # One-qubit state=32 bytes. Three distinct rows need 96 bytes,
    # including the separate Y set. Duplicates do not retain extra states.
    kernel = ry_kernel(max_statevector_bytes=64)
    with pytest.raises(ValueError, match="max_statevector_bytes"):
        kernel.evaluate([[0], [1]], [[2]])
    np.testing.assert_allclose(
        kernel.evaluate([[0], [0]], [[1]]), analytic_ry([[0], [0]], [[1]]), atol=1e-12
    )


def test_one_core_state_per_distinct_row_and_no_dense_matrix(monkeypatch):
    from cqlib_algorithms.operators.quantum_kernel import (
        fidelity_quantum_kernel as module,
    )

    original = module.Statevector.from_circuit
    calls = []

    def count(circuit):
        calls.append(circuit)
        return original(circuit)

    monkeypatch.setattr(module.Statevector, "from_circuit", count)
    monkeypatch.setattr(
        Circuit, "to_matrix", lambda *a: pytest.fail("no 4**n allocation")
    )
    ry_kernel().evaluate([[0], [1], [0]], [[2], [1]])
    assert len(calls) == 3


@pytest.mark.parametrize(
    "value,valid",
    [
        (-1e-13, True),
        (1 + 1e-13, True),
        (-1e-4, False),
        (1.001, False),
        (math.nan, False),
    ],
)
def test_exact_roundoff_clipping_only(value, valid, monkeypatch):
    monkeypatch.setattr(
        "cqlib_algorithms.operators.quantum_kernel.fidelity_quantum_kernel.state_fidelity_pure",
        lambda *a: value,
    )
    if valid:
        assert ry_kernel().evaluate([[0]], [[1]])[0, 0] == min(1, max(0, value))
    else:
        with pytest.raises(ValueError, match="Core fidelity"):
            ry_kernel().evaluate([[0]], [[1]])


def test_public_operator_exports_share_one_implementation():
    import importlib.util

    from cqlib_algorithms import algorithms
    from cqlib_algorithms.operators import FidelityQuantumKernel as operator_export

    assert operator_export is FidelityQuantumKernel
    assert not hasattr(algorithms, "FidelityQuantumKernel")
    assert (
        importlib.util.find_spec("cqlib_algorithms.algorithms.quantum_kernel") is None
    )
    assert FidelityQuantumKernel.__module__ == (
        "cqlib_algorithms.operators.quantum_kernel.fidelity_quantum_kernel"
    )


def test_quantum_kernel_operator_has_no_direct_algorithm_dependency():
    from cqlib_algorithms.operators import quantum_kernel

    package_name = quantum_kernel.__name__
    package_path = Path(quantum_kernel.__file__).parent
    imported_names = []
    for path in package_path.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                if node.level:
                    name = importlib.util.resolve_name(name, package_name)
                imported_names.append(name)
                imported_names.extend(name + "." + alias.name for alias in node.names)
    assert all(
        name != "cqlib_algorithms.algorithms"
        and not name.startswith("cqlib_algorithms.algorithms.")
        for name in imported_names
    ), imported_names
