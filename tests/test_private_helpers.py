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

"""Contract tests for project-wide private validation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from cqlib import Circuit, Parameter, Qubit
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms._utils.circuit import (
    snapshot_bound_static_unitary_circuit,
    validate_bound_static_unitary_circuit,
    validate_static_unitary_structure,
)
from cqlib_algorithms._utils.hamiltonian import (
    aggregate_finite_pauli_coefficients,
    aggregate_finite_pauli_terms,
    snapshot_nonzero_hermitian_hamiltonian,
    validate_hermitian_observable,
    validate_nonzero_hermitian_hamiltonian,
)
from cqlib_algorithms._utils.numeric import (
    copy_finite_real_array,
    normalize_finite_real,
    normalize_nonnegative_real,
    normalize_positive_integer,
    validate_nonnegative_integer,
    validate_optional_positive_integer,
    validate_optional_seed,
    validate_positive_integer,
    validate_positive_qubit_count,
    validate_repetitions,
)
from cqlib_algorithms._utils.parameters import (
    snapshot_parameter_bindings,
    validate_parameter_prefix,
)


def test_numeric_helpers_preserve_strict_types_and_defensive_arrays() -> None:
    values = [[1, 2], [3, 4]]
    snapshot = copy_finite_real_array(values, "values", 2)
    values[0][0] = 9

    np.testing.assert_array_equal(snapshot, [[1.0, 2.0], [3.0, 4.0]])
    assert normalize_positive_integer(2, "count") == 2
    assert normalize_finite_real(np.float64(0.25), "value") == 0.25
    assert normalize_nonnegative_real(0, "value") == 0.0
    assert validate_positive_integer(1, "count") is None
    assert validate_optional_positive_integer(None, "count") is None
    assert validate_optional_positive_integer(1, "count") is None
    assert validate_optional_seed(None) is None
    assert validate_nonnegative_integer(0, name="count") is None
    assert validate_positive_qubit_count(1) is None
    assert validate_repetitions(0) is None

    with pytest.raises(TypeError, match="count must be an integer"):
        validate_positive_integer(True, "count")
    with pytest.raises(TypeError, match="count must be an integer or None"):
        validate_optional_positive_integer(True, "count")
    with pytest.raises(ValueError, match="value must contain finite values"):
        copy_finite_real_array([np.inf], "value", 1)


def test_parameter_helpers_validate_and_snapshot_without_coercing_names() -> None:
    bindings = {"theta": 0.25}
    snapshot = snapshot_parameter_bindings(bindings, name="parameter_bindings")
    bindings["theta"] = 0.5

    assert dict(snapshot) == {"theta": 0.25}
    assert validate_parameter_prefix("") is None
    with pytest.raises(TypeError, match="prefix must be a string"):
        validate_parameter_prefix(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parameter binding values"):
        snapshot_parameter_bindings({"theta": True}, name="parameter_bindings")


def test_circuit_snapshot_normalizes_qubits_and_is_defensive() -> None:
    source = Circuit([7, 2])
    source.h(Qubit(7))
    source.cx(Qubit(7), Qubit(2))

    snapshot = snapshot_bound_static_unitary_circuit(source, "source")
    expected = np.array(snapshot.to_matrix(), copy=True)
    source.x(Qubit(7))

    assert [qubit.id for qubit in snapshot.qubits] == [0, 1]
    np.testing.assert_allclose(snapshot.to_matrix(), expected, rtol=0.0, atol=1e-12)
    assert validate_static_unitary_structure(snapshot, label="source") is None
    assert validate_bound_static_unitary_circuit(snapshot, "source") is None


def test_bound_circuit_validator_rejects_active_parameters() -> None:
    circuit = Circuit(1)
    circuit.ry(0, Parameter("theta"))

    with pytest.raises(ValueError, match="circuit must be fully bound"):
        validate_bound_static_unitary_circuit(circuit, "circuit")


def test_hamiltonian_helpers_aggregate_phase_and_duplicate_terms() -> None:
    first = PauliString.from_str("Z")
    first.phase = Phase.i()
    second = PauliString.from_str("Z")
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(first, -1.0j)
    hamiltonian.add_term(second, 2.0)

    coefficients = aggregate_finite_pauli_coefficients(
        hamiltonian,
        name="hamiltonian",
    )
    terms = aggregate_finite_pauli_terms(hamiltonian, name="hamiltonian")

    assert coefficients[(1, 0, 1)] == 3.0
    assert len(terms) == 1
    assert terms[0][0].phase == Phase.plus()
    assert terms[0][1] == 3.0
    assert validate_hermitian_observable(hamiltonian, name="hamiltonian") is None


def test_nonzero_hamiltonian_snapshot_is_defensive_and_rejects_cancellation() -> None:
    original = Hamiltonian.from_pauli(PauliString.from_str("X"))
    snapshot = snapshot_nonzero_hermitian_hamiltonian(
        original,
        name="hamiltonian",
    )
    expected = np.array(snapshot.to_matrix(), copy=True)
    original.add_term(PauliString.from_str("Z"), 4.0)

    np.testing.assert_allclose(snapshot.to_matrix(), expected, rtol=0.0, atol=1e-12)

    cancelled = Hamiltonian(1)
    cancelled.add_term(PauliString.from_str("Z"), 1.0)
    cancelled.add_term(PauliString.from_str("Z"), -1.0)
    assert validate_hermitian_observable(cancelled, name="hamiltonian") is None
    with pytest.raises(ValueError, match="must not be empty after simplification"):
        validate_nonzero_hermitian_hamiltonian(cancelled, name="hamiltonian")


def test_hamiltonian_validator_rejects_nonhermitian_and_nonfinite_terms() -> None:
    nonhermitian = Hamiltonian(1)
    nonhermitian.add_term(PauliString.from_str("X"), 1.0j)
    with pytest.raises(ValueError, match="hamiltonian must be Hermitian"):
        validate_hermitian_observable(nonhermitian, name="hamiltonian")

    nonfinite = Hamiltonian(1)
    nonfinite.add_term(PauliString.from_str("X"), np.inf)
    with pytest.raises(ValueError, match="hamiltonian coefficients must be finite"):
        validate_hermitian_observable(nonfinite, name="hamiltonian")


def test_observable_validator_handles_standalone_pauli_phases() -> None:
    observable = PauliString.from_str("Z")
    assert validate_hermitian_observable(observable, name="observable") is None

    observable.phase = Phase.minus()
    assert validate_hermitian_observable(observable, name="observable") is None

    observable.phase = Phase.i()
    with pytest.raises(ValueError, match=r"phase must be \+1 or -1"):
        validate_hermitian_observable(observable, name="observable")
