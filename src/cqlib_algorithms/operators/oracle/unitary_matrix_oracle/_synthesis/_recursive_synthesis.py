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

"""Recursive Block-ZXZ synthesis for unitary-matrix oracles."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from cqlib import Circuit, Qubit
from cqlib.compile.transform.decompose.unitary import (
    synthesize_numeric_1q_unitary,
    synthesize_numeric_2q_unitary,
)

from ....arithmetic.amplitude_encoded_rotation.multiplexed import UCRZ
from ._block_factors import factor_block_zxz
from ._common import (
    assert_square_unitary,
    is_diagonal,
    is_unitary,
    num_qubits_from_matrix,
    project_to_unitary,
    safe_project_subunitary,
)
from ._demultiplex import demultiplex_block_zxz


def is_base_case_unitary(
    unitary: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> bool:
    """Return whether ``unitary`` is a one- or two-qubit synthesis leaf."""

    assert_square_unitary(unitary, atol=atol)
    return unitary.shape[0] in (2, 4)


def build_base_case_circuit(
    unitary: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> Circuit:
    """Return a circuit for a one- or two-qubit unitary leaf."""

    assert_square_unitary(unitary, atol=atol)
    num_qubits = num_qubits_from_matrix(unitary)
    projected_unitary = project_to_unitary(unitary)

    if num_qubits == 1:
        decomposition = synthesize_numeric_1q_unitary(projected_unitary)
        circuit = Circuit(1)
        circuit.u(
            0,
            decomposition.theta,
            decomposition.phi,
            decomposition.lambda_,
        )
        circuit.set_global_phase(decomposition.global_phase)
        return circuit

    if num_qubits == 2:
        synthesis = synthesize_numeric_2q_unitary(projected_unitary, 1, Qubit(0))
        circuit = Circuit.from_operations([Qubit(0), Qubit(1)], synthesis.operations)
        circuit.set_global_phase(synthesis.global_phase)
        return circuit

    raise ValueError("base-case synthesis only supports one or two qubits")


def collect_recursive_subunitaries(
    demultiplexed_factors: dict[str, tuple[npt.NDArray[np.complex128], ...]],
) -> dict[str, npt.NDArray[np.complex128]]:
    """Return the four recursive child unitaries from one split layer."""

    a_branch = demultiplexed_factors["a_branch"]
    c_branch = demultiplexed_factors["c_branch"]
    middle_branch = demultiplexed_factors["middle_branch"]
    return {
        "va": np.asarray(a_branch[0], dtype=np.complex128),
        "vb": np.asarray(middle_branch[0], dtype=np.complex128),
        "wb": np.asarray(middle_branch[2], dtype=np.complex128),
        "wc": np.asarray(c_branch[2], dtype=np.complex128),
    }


def extract_diagonal_rotation_table(
    diagonal_factor: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> npt.NDArray[np.float64]:
    """Return the UCRZ angle table encoded by a diagonal unitary factor."""

    if diagonal_factor.ndim != 2:
        raise ValueError("diagonal_factor must be a two-dimensional matrix")
    if diagonal_factor.shape[0] != diagonal_factor.shape[1]:
        raise ValueError("diagonal_factor must be square")
    if not is_diagonal(diagonal_factor, atol=atol):
        raise ValueError("diagonal_factor must be diagonal")
    if not is_unitary(diagonal_factor, atol=atol):
        raise ValueError("diagonal_factor must be unitary")

    angles = -2.0 * np.angle(np.diag(diagonal_factor))
    return np.asarray(angles, dtype=np.float64)


def build_diagonal_rotation_circuit(
    diagonal_factor: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> Circuit:
    """Return the local UCRZ circuit associated with ``diagonal_factor``."""

    angles = extract_diagonal_rotation_table(diagonal_factor, atol=atol)
    num_controls = int(math.log2(angles.size))
    return UCRZ(num_controls, angles).build_circuit()


def build_diagonal_unitary_circuit(
    diagonal_unitary: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> Circuit:
    """Return a circuit for an ``n``-qubit diagonal unitary.

    The phase table is synthesized level by level using diagonal RZ
    multiplexors. This avoids routing already diagonal inputs through the full
    Block-ZXZ demultiplexing path.
    """

    if diagonal_unitary.ndim != 2:
        raise ValueError("diagonal_unitary must be a two-dimensional matrix")
    if diagonal_unitary.shape[0] != diagonal_unitary.shape[1]:
        raise ValueError("diagonal_unitary must be square")
    if not is_diagonal(diagonal_unitary, atol=atol):
        raise ValueError("diagonal_unitary must be diagonal")
    if not is_unitary(diagonal_unitary, atol=atol):
        raise ValueError("diagonal_unitary must be unitary")

    num_qubits = num_qubits_from_matrix(diagonal_unitary)
    phases = np.angle(np.diag(diagonal_unitary))
    circuit = Circuit(num_qubits)
    circuit.set_global_phase(float(np.sum(phases) / phases.size))

    # Match the phase-table recursion used for diagonal RZ multiplexors in
    # state synthesis: each level encodes block-average phase differences.
    for level in range(1, num_qubits + 1):
        half = 1 << (level - 1)
        angles: list[float] = []
        for block in range(1 << (num_qubits - level)):
            start = block << level
            lower_phase_sum = float(np.sum(phases[start : start + half]))
            upper_phase_sum = float(
                np.sum(phases[start + half : start + 2 * half]),
            )
            angles.append((upper_phase_sum - lower_phase_sum) / half)

        controls = list(range(level, num_qubits))
        target = level - 1
        local_circuit = UCRZ(len(controls), angles).build_circuit()
        circuit.compose(local_circuit, [*controls, target])

    return circuit


def assemble_block_zxz_layer(
    *,
    num_qubits: int,
    va_child: Circuit,
    vb_child: Circuit,
    wb_child: Circuit,
    wc_child: Circuit,
    a_circuit: Circuit,
    b_circuit: Circuit,
    c_circuit: Circuit,
) -> Circuit:
    """Assemble one recursive Block-ZXZ layer from child circuits."""

    circuit = Circuit(num_qubits)
    child_qubits = list(range(num_qubits - 1))
    top_qubit = num_qubits - 1

    circuit.compose(wc_child, child_qubits)
    circuit.compose(c_circuit)
    circuit.h(top_qubit)
    circuit.compose(wb_child, child_qubits)
    circuit.compose(b_circuit)
    circuit.compose(vb_child, child_qubits)
    circuit.h(top_qubit)
    circuit.compose(a_circuit)
    circuit.compose(va_child, child_qubits)

    return circuit


def synthesize_unitary_matrix_circuit(
    unitary: npt.NDArray[np.complex128],
    *,
    atol: float,
) -> Circuit:
    """Recursively synthesize ``unitary`` with the Block-ZXZ construction."""

    normalized = np.asarray(unitary, dtype=np.complex128)
    assert_square_unitary(normalized, atol=atol)

    if is_base_case_unitary(normalized, atol=atol):
        return build_base_case_circuit(normalized, atol=atol)
    if is_diagonal(normalized, atol=atol):
        return build_diagonal_unitary_circuit(normalized, atol=atol)

    num_qubits = num_qubits_from_matrix(normalized)
    a1_factor, a2_factor, b_factor, c_factor = factor_block_zxz(
        normalized,
        atol=atol,
    )
    demultiplexed_factors = demultiplex_block_zxz(
        a1_factor,
        a2_factor,
        b_factor,
        c_factor,
        atol=atol,
    )
    subunitaries = collect_recursive_subunitaries(demultiplexed_factors)
    subunitaries = {
        key: safe_project_subunitary(
            value,
            atol=atol,
            label=f"recursive subunit {key}",
        )
        for key, value in subunitaries.items()
    }

    return assemble_block_zxz_layer(
        num_qubits=num_qubits,
        va_child=synthesize_unitary_matrix_circuit(subunitaries["va"], atol=atol),
        vb_child=synthesize_unitary_matrix_circuit(subunitaries["vb"], atol=atol),
        wb_child=synthesize_unitary_matrix_circuit(subunitaries["wb"], atol=atol),
        wc_child=synthesize_unitary_matrix_circuit(subunitaries["wc"], atol=atol),
        a_circuit=build_diagonal_rotation_circuit(
            demultiplexed_factors["a_branch"][1],
            atol=atol,
        ),
        b_circuit=build_diagonal_rotation_circuit(
            demultiplexed_factors["middle_branch"][1],
            atol=atol,
        ),
        c_circuit=build_diagonal_rotation_circuit(
            demultiplexed_factors["c_branch"][1],
            atol=atol,
        ),
    )
