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
"""Shared internal helpers used only by the adder test module.

This module is intentionally restricted to internal helper utilities. It
exposes no public API and should never be imported outside of the adder
tests located in test_adders.py.

All helpers rely exclusively on the cqlib quantum layer and NumPy, in line
with the project rule that testing must not pull in external quantum
computing frameworks unless an explicit difference comparison is requested.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from cqlib.qis.state import Statevector

from cqlib import Circuit
from cqlib_algorithms.operators.arithmetic import (
    FullAdder,
    HalfAdder,
    ModularAdder,
)

AdderT = HalfAdder | FullAdder | ModularAdder


def _apply_x_to_value(
    circuit: Circuit,
    value: int,
    bit_indices: Iterable[int],
) -> None:
    """Flip qubits on circuit at bit_indices to encode integer value.

    The integer value is interpreted with the standard LSB-first convention
    of the arithmetic library. The first element of bit_indices holds the
    least significant bit of value. Any bit beyond width of bit_indices
    must be zero; a ValueError is raised otherwise.
    """
    positions = list(bit_indices)
    width = len(positions)
    if not 0 <= value < (1 << width):
        raise ValueError(
            f"Value {value} does not fit into a {width}-bit register."
        )
    for local_index, qubit in enumerate(positions):
        if (value >> local_index) & 1:
            circuit.x(qubit)


def _encode_adder_input_circuit(
    adder: AdderT,
    a: int,
    b: int,
    carry_in: int = 0,
) -> Circuit:
    """Build a circuit that encodes a b carry_in then runs the adder."""
    encoded = Circuit(adder.num_qubits)
    _apply_x_to_value(encoded, a, adder.a_register_indices)
    _apply_x_to_value(encoded, b, adder.b_register_indices)
    if isinstance(adder, FullAdder) and carry_in:
        encoded.x(adder.carry_in_index)
    encoded.compose(adder.build_circuit(), list(range(adder.num_qubits)))
    return encoded


def _deterministic_measurement_outcome(sv: Statevector) -> int:
    """Return the index of the unique non-zero amplitude of sv."""
    probs = np.asarray(sv.probabilities(), dtype=np.float64)
    if not np.isclose(probs.max(), 1.0, atol=1e-8):
        raise AssertionError(
            f"Expected deterministic state with max prob 1, got {probs.max()}"
        )
    return int(np.argmax(probs))


def _decode_subregister_bits(
    outcome_int: int,
    bit_indices: Iterable[int],
) -> int:
    """Decode integer stored in bit_indices from outcome_int LSB first."""
    result = 0
    for shift, qubit in enumerate(bit_indices):
        if (outcome_int >> qubit) & 1:
            result |= 1 << shift
    return result


def _decode_adder_outcome(
    adder: AdderT,
    outcome_int: int,
) -> dict[str, int]:
    """Decode all registers of adder from a deterministic outcome integer."""
    decoded: dict[str, int] = {
        "a": _decode_subregister_bits(outcome_int, adder.a_register_indices),
        "b_sum": _decode_subregister_bits(
            outcome_int, adder.b_register_indices
        ),
    }
    if isinstance(adder, (HalfAdder, FullAdder)):
        decoded["carry_out"] = (outcome_int >> adder.carry_out_index) & 1
    if isinstance(adder, FullAdder):
        decoded["carry_in"] = (outcome_int >> adder.carry_in_index) & 1
    return decoded


def _run_adder(
    adder: AdderT,
    a: int,
    b: int,
    carry_in: int = 0,
) -> dict[str, int]:
    """Encode, simulate and decode a single deterministic adder run."""
    circuit = _encode_adder_input_circuit(adder, a, b, carry_in=carry_in)
    statevector = Statevector.from_circuit(circuit)
    outcome = _deterministic_measurement_outcome(statevector)
    return _decode_adder_outcome(adder, outcome)


def _collect_register_index_lists(adder: AdderT) -> list[list[int]]:
    """Return every index list exposed by adder for sanity checks."""
    index_groups: list[list[int]] = [
        list(adder.a_register_indices),
        list(adder.b_register_indices),
    ]
    if isinstance(adder, FullAdder):
        index_groups.append([adder.carry_in_index])
    if isinstance(adder, (HalfAdder, FullAdder)):
        index_groups.append([adder.carry_out_index])
    helper_index = getattr(adder, "helper_index", None)
    if helper_index is not None:
        index_groups.append([helper_index])
    return index_groups


def _assert_clean_register_layout(adder: AdderT) -> None:
    """Assert no overlap and every index inside [0, num_qubits)."""
    groups = _collect_register_index_lists(adder)
    flat: list[int] = [idx for group in groups for idx in group]
    assert len(flat) == len(set(flat)), (
        f"Register indices overlap for {type(adder).__name__}: {flat}"
    )
    assert all(0 <= idx < adder.num_qubits for idx in flat), (
        f"Index out of range for {type(adder).__name__}: {flat}"
    )


def _expected_num_qubits(
    adder_cls: type[AdderT],
    num_state_qubits: int,
) -> int:
    """Return the canonical total qubit width for an adder class."""
    n = num_state_qubits
    if adder_cls is HalfAdder:
        return 2 * n + 2
    if adder_cls is FullAdder:
        return 2 * n + 2
    if adder_cls is ModularAdder:
        return 2 * n + 1
    raise TypeError(f"Unknown adder class: {adder_cls!r}")

