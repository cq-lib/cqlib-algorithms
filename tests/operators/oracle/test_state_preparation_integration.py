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

"""Integration tests for state-preparation-backed reflections."""

from __future__ import annotations

import copy

import numpy as np

from cqlib import Circuit, Parameter
from cqlib_algorithms.operators.oracle import GeneralStateReflection
from cqlib_algorithms.operators.state_preparation import (
    UniformSuperposition,
    UserCircuitPreparation,
)


def test_uniform_superposition_adapter_matches_direct_circuit_reflection():
    preparation = UniformSuperposition(3)
    direct_circuit = preparation.build_circuit("direct")

    adapted = GeneralStateReflection.from_state_preparation(
        preparation, prefix="uniform"
    ).build_circuit()
    direct = GeneralStateReflection(direct_circuit).build_circuit()

    np.testing.assert_allclose(
        adapted.to_matrix(), direct.to_matrix(), rtol=0.0, atol=1e-12
    )


def test_user_circuit_adapter_preserves_prefix_snapshot_and_global_phase():
    theta = Parameter("theta")
    source = Circuit([7, 2])
    source.ry(7, theta)
    source.cx(7, 2)
    source.set_global_phase(0.417)
    source_snapshot = copy.deepcopy(source)
    preparation = UserCircuitPreparation(source)
    prepared_snapshot = preparation.build_circuit("before")

    adapted_component = GeneralStateReflection.from_state_preparation(
        preparation, prefix="adapter_prefix"
    )
    direct_component = GeneralStateReflection(prepared_snapshot)
    source.x(7)
    prepared_snapshot.x(0)

    assert adapted_component.state_preparation.used_symbols == ["theta"]
    value = 0.638
    normalized_source = Circuit(2)
    normalized_source.compose(source_snapshot, [0, 1])
    np.testing.assert_allclose(
        adapted_component.state_preparation.assign_parameters(
            {"theta": value}
        ).to_matrix(),
        normalized_source.assign_parameters({"theta": value}).to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adapted_component.build_circuit()
        .assign_parameters({"theta": value})
        .to_matrix(),
        direct_component.build_circuit()
        .assign_parameters({"theta": value})
        .to_matrix(),
        rtol=0.0,
        atol=1e-12,
    )
