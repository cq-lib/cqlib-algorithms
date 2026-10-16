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

"""Private numerical validation and state-synthesis routines.

References
----------
[1] M. Möttönen, J. J. Vartiainen, V. Bergholm, and M. M. Salomaa,
    "Transformation of quantum states using uniformly controlled rotations,"
    Quantum Information and Computation, 5(6), 467-473, 2005.
    DOI: 10.26421/QIC5.6-5.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import TypeAlias

from cqlib.circuit import Circuit
from cqlib.circuit.gates import StandardGate

ComplexVector: TypeAlias = tuple[complex, ...]
SparseEntry: TypeAlias = tuple[int, complex]

_NORMALIZATION_TOLERANCE = 1e-12


def _component_scale(values: Iterable[complex]) -> float:
    """Return the largest finite real or imaginary component magnitude."""
    return max(max(abs(value.real), abs(value.imag)) for value in values)


def _stable_norm(values: Iterable[complex]) -> float:
    """Return a finite-vector norm without squaring underflow or overflow."""
    norm = 0.0
    for value in values:
        norm = math.hypot(norm, abs(value))
    return norm


def validate_positive_qubit_count(num_qubits: int) -> None:
    """Validate a strictly positive qubit count."""
    if type(num_qubits) is not int:
        raise TypeError("num_qubits must be an int")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")


def validate_prefix(prefix: str) -> None:
    """Validate a component parameter-name prefix."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str")


def copy_complex_vector(values: Sequence[complex], *, name: str) -> ComplexVector:
    """Copy a finite, power-of-two vector with at least two entries."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")

    copied: list[complex] = []
    for index, value in enumerate(values):
        try:
            item = complex(value)
        except TypeError as error:
            raise TypeError(
                f"{name}[{index}] must be convertible to complex"
            ) from error
        except ValueError as error:
            raise ValueError(f"{name}[{index}] is not a valid complex value") from error
        if not math.isfinite(item.real) or not math.isfinite(item.imag):
            raise ValueError(f"{name}[{index}] must be finite")
        copied.append(item)

    length = len(copied)
    if length < 2:
        raise ValueError(f"{name} must contain at least two values")
    if length & (length - 1):
        raise ValueError(f"the length of {name} must be a power of two")
    return tuple(copied)


def normalize_nonzero(values: ComplexVector, *, name: str) -> ComplexVector:
    """Normalize a nonzero vector without overflowing on large finite inputs."""
    scale = _component_scale(values)
    if scale == 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    scaled = tuple(value / scale for value in values)
    scaled_norm = _stable_norm(scaled)
    return tuple(value / scaled_norm for value in scaled)


def normalize_unit_vector(values: ComplexVector, *, name: str) -> ComplexVector:
    """Validate a unit vector and correct accepted roundoff in a new tuple."""
    scale = _component_scale(values)
    if scale == 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    scaled = tuple(value / scale for value in values)
    scaled_norm = _stable_norm(scaled)
    norm = scale * scaled_norm
    if not math.isfinite(norm) or not math.isclose(
        norm,
        1.0,
        rel_tol=_NORMALIZATION_TOLERANCE,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"{name} must have unit norm within 1e-12")
    return tuple(value / scaled_norm for value in scaled)


def validate_normalized_vector(values: ComplexVector, *, name: str) -> None:
    """Validate that an existing finite vector has unit norm within 1e-12."""
    scale = _component_scale(values)
    if scale == 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    scaled_norm = _stable_norm(value / scale for value in values)
    norm = scale * scaled_norm
    if not math.isfinite(norm) or not math.isclose(
        norm,
        1.0,
        rel_tol=_NORMALIZATION_TOLERANCE,
        abs_tol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"{name} must have unit norm within 1e-12")


def build_mottonen_circuit(amplitudes: ComplexVector) -> Circuit:
    """Build exact Möttönen state preparation in Cqlib little-endian order.

    Magnitudes are synthesized by uniformly controlled RY rotations. Relative
    phases are synthesized by uniformly controlled RZ rotations, with the mean
    input phase stored as the circuit global phase. Each uniformly controlled
    rotation normally lowers through a cyclic Gray code and a Walsh-Hadamard
    angle transform using one-qubit rotations and CX gates. Numerically extreme
    angle ranges fall back to mutually exclusive controlled rotations so that
    representable nonzero angles are not lost to cancellation.
    """
    num_qubits = len(amplitudes).bit_length() - 1
    circuit = Circuit(num_qubits)

    # Möttönen magnitude angles. At level k, each contiguous 2**k block
    # corresponds to one assignment of the more-significant control qubits.
    for k in range(num_qubits, 0, -1):
        half = 1 << (k - 1)
        controls = tuple(range(k, num_qubits))
        angles: list[float] = []
        for block in range(1 << (num_qubits - k)):
            start = block << k
            lower_norm = _stable_norm(amplitudes[i] for i in range(start, start + half))
            upper_norm = _stable_norm(
                amplitudes[i] for i in range(start + half, start + 2 * half)
            )
            if lower_norm == 0.0 and upper_norm == 0.0:
                angles.append(0.0)
            elif lower_norm < upper_norm:
                # RY(-2*atan(lower/upper)) X |0> produces positive
                # amplitudes (lower, upper) without subtracting a tiny angle
                # from pi. The X is conditioned on this block's controls.
                _append_conditioned_gate(
                    circuit,
                    StandardGate.X,
                    target=k - 1,
                    controls=controls,
                    basis_pattern=start,
                )
                angles.append(-2.0 * math.atan2(lower_norm, upper_norm))
            else:
                angles.append(2.0 * math.atan2(upper_norm, lower_norm))
        _append_uniformly_controlled_rotation(
            circuit,
            StandardGate.RY,
            target=k - 1,
            controls=controls,
            conditional_angles=angles,
        )

    phases = tuple(cmath.phase(value) if value != 0.0 else 0.0 for value in amplitudes)
    circuit.set_global_phase(math.fsum(phases) / len(phases))

    # The phase angles are block-average differences. Diagonal RZ multiplexors
    # commute, so applying levels from least to most significant is convenient.
    for k in range(1, num_qubits + 1):
        half = 1 << (k - 1)
        angles = []
        for block in range(1 << (num_qubits - k)):
            start = block << k
            lower_phase_sum = math.fsum(phases[start : start + half])
            upper_phase_sum = math.fsum(phases[start + half : start + 2 * half])
            angles.append((upper_phase_sum - lower_phase_sum) / half)
        _append_uniformly_controlled_rotation(
            circuit,
            StandardGate.RZ,
            target=k - 1,
            controls=tuple(range(k, num_qubits)),
            conditional_angles=angles,
        )

    return circuit


def canonicalize_sparse_entries(
    num_qubits: int, entries: Sequence[tuple[int, complex]]
) -> tuple[SparseEntry, ...]:
    """Validate, copy, normalize, and sort sparse amplitudes."""
    validate_positive_qubit_count(num_qubits)
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("entries must be a sequence")
    if not entries:
        raise ValueError("entries must not be empty")

    maximum_index = 1 << num_qubits
    copied: list[SparseEntry] = []
    seen: set[int] = set()
    for position, entry in enumerate(entries):
        if (
            isinstance(entry, (str, bytes))
            or not isinstance(entry, Sequence)
            or len(entry) != 2
        ):
            raise TypeError(f"entries[{position}] must be an (index, amplitude) pair")
        index, raw_amplitude = entry
        if type(index) is not int:
            raise TypeError(f"entries[{position}][0] must be an int")
        if index < 0 or index >= maximum_index:
            raise ValueError(f"entries[{position}][0] must be in [0, {maximum_index})")
        if index in seen:
            raise ValueError(f"duplicate sparse index {index}")
        seen.add(index)
        try:
            amplitude = complex(raw_amplitude)
        except TypeError as error:
            raise TypeError(
                f"entries[{position}][1] must be convertible to complex"
            ) from error
        except ValueError as error:
            raise ValueError(
                f"entries[{position}][1] is not a valid complex value"
            ) from error
        if not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag):
            raise ValueError(f"entries[{position}][1] must be finite")
        if amplitude == 0.0:
            raise ValueError(f"entries[{position}][1] must not be zero")
        copied.append((index, amplitude))

    scale = _component_scale(amplitude for _, amplitude in copied)
    scaled = [(index, amplitude / scale) for index, amplitude in copied]
    scaled_norm = _stable_norm(amplitude for _, amplitude in scaled)
    normalized = [(index, amplitude / scaled_norm) for index, amplitude in scaled]
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def validate_sparse_entries(num_qubits: int, entries: tuple[SparseEntry, ...]) -> None:
    """Validate an existing canonical sparse-amplitude snapshot."""
    validate_positive_qubit_count(num_qubits)
    if not entries:
        raise ValueError("entries must not be empty")

    maximum_index = 1 << num_qubits
    previous_index = -1
    amplitudes: list[complex] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"entries[{position}] must be an (index, amplitude) pair")
        index, amplitude = entry
        if type(index) is not int:
            raise TypeError(f"entries[{position}][0] must be an int")
        if index < 0 or index >= maximum_index:
            raise ValueError(f"entries[{position}][0] must be in [0, {maximum_index})")
        if index <= previous_index:
            raise ValueError("entries must have unique indices in ascending order")
        previous_index = index
        if not isinstance(amplitude, complex):
            raise TypeError(f"entries[{position}][1] must be a complex value")
        if not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag):
            raise ValueError(f"entries[{position}][1] must be finite")
        if amplitude == 0.0:
            raise ValueError(f"entries[{position}][1] must not be zero")
        amplitudes.append(amplitude)

    validate_normalized_vector(tuple(amplitudes), name="entries")


def build_sparse_two_level_circuit(
    num_qubits: int, entries: tuple[SparseEntry, ...]
) -> Circuit:
    """Prepare sparse amplitudes using exact support-set two-level synthesis.

    The routine stores only the supplied support. It starts in a maximum-
    magnitude support basis state, which preserves representable tiny relative
    amplitudes, and allocates every other amplitude with a two-level Givens
    unitary. A Gray path conjugates that unitary into a multi-controlled U gate
    between adjacent computational basis states. No dense length-2**n vector or
    dense unitary is constructed.
    """
    circuit = Circuit(num_qubits)
    reference, reference_amplitude = max(entries, key=lambda entry: abs(entry[1]))
    for qubit in range(num_qubits):
        if (reference >> qubit) & 1:
            circuit.x(qubit)

    other_entries = tuple(entry for entry in entries if entry[0] != reference)
    suffix_norms = [0.0] * (len(other_entries) + 1)
    suffix_norms[-1] = abs(reference_amplitude)
    for position in range(len(other_entries) - 1, -1, -1):
        suffix_norms[position] = math.hypot(
            abs(other_entries[position][1]), suffix_norms[position + 1]
        )

    for position, (basis_index, amplitude) in enumerate(other_entries):
        after = suffix_norms[position + 1]
        theta = 2.0 * math.atan2(abs(amplitude), after)
        phase = cmath.phase(amplitude)
        _append_two_level_allocation(
            circuit,
            num_qubits=num_qubits,
            reference=reference,
            destination=basis_index,
            theta=theta,
            phase=phase,
        )

    reference_phase = cmath.phase(reference_amplitude)
    if reference_phase != 0.0:
        _append_basis_state_phase(
            circuit,
            num_qubits=num_qubits,
            basis_index=reference,
            phase=reference_phase,
        )
    return circuit


def _append_uniformly_controlled_rotation(
    circuit: Circuit,
    rotation_gate: StandardGate,
    *,
    target: int,
    controls: tuple[int, ...],
    conditional_angles: Sequence[float],
) -> None:
    """Append a Gray-code uniformly controlled RY or RZ rotation."""
    control_count = len(controls)
    expected_angles = 1 << control_count
    if len(conditional_angles) != expected_angles:
        raise ValueError("uniformly controlled rotation angle count is inconsistent")
    if all(angle == 0.0 for angle in conditional_angles):
        return
    if control_count == 0:
        angle = conditional_angles[0]
        if angle != 0.0:
            circuit.append_gate(rotation_gate(angle), [target])
        return

    nonzero_angles = [abs(angle) for angle in conditional_angles if angle != 0.0]
    if nonzero_angles and (
        min(nonzero_angles) < max(nonzero_angles) * 1e-14
        or min(nonzero_angles) / expected_angles == 0.0
    ):
        # A Walsh transform would combine tiny and order-one angles and lose
        # the tiny values to floating-point cancellation. Direct orthogonal
        # control patterns preserve every representable nonzero angle.
        for pattern, angle in enumerate(conditional_angles):
            if angle == 0.0:
                continue
            basis_pattern = sum(
                ((pattern >> position) & 1) << qubit
                for position, qubit in enumerate(controls)
            )
            _append_conditioned_gate(
                circuit,
                rotation_gate(angle),
                target=target,
                controls=controls,
                basis_pattern=basis_pattern,
            )
        return

    gray_codes = tuple(index ^ (index >> 1) for index in range(expected_angles))
    transformed_angles = tuple(
        math.fsum(
            (
                -conditional_angles[pattern]
                if (pattern & gray_codes[rotation]).bit_count() & 1
                else conditional_angles[pattern]
            )
            for pattern in range(expected_angles)
        )
        / expected_angles
        for rotation in range(expected_angles)
    )

    for rotation, angle in enumerate(transformed_angles):
        if angle != 0.0:
            circuit.append_gate(rotation_gate(angle), [target])
        changed_bit = (
            gray_codes[rotation] ^ gray_codes[(rotation + 1) % expected_angles]
        )
        circuit.cx(controls[changed_bit.bit_length() - 1], target)


def _append_two_level_allocation(
    circuit: Circuit,
    *,
    num_qubits: int,
    reference: int,
    destination: int,
    theta: float,
    phase: float,
) -> None:
    """Mix only two requested basis states with an exact complex Givens gate.

    The selected U parameters satisfy ``phi + lambda == 0 (mod 2*pi)``.
    Therefore the controlled one-qubit block is an SU(2) Givens rotation: its
    reference column assigns ``exp(1j * phase) * sin(theta / 2)`` to the
    destination and retains real ``cos(theta / 2)`` at the reference. This
    invariant is specific to allocation and is not an arbitrary controlled-U
    synthesis claim.
    """
    differing_bits = tuple(
        qubit for qubit in range(num_qubits) if ((reference ^ destination) >> qubit) & 1
    )
    if not differing_bits:
        raise ValueError("two-level endpoints must be distinct")

    path = [reference]
    current = reference
    for qubit in differing_bits[:-1]:
        current ^= 1 << qubit
        path.append(current)

    for first, second in pairwise(path):
        _append_adjacent_basis_swap(circuit, num_qubits, first, second)

    mapped_reference = path[-1]
    target = differing_bits[-1]
    if (mapped_reference >> target) & 1:
        lambda_angle = phase + math.pi
        phi_angle = -lambda_angle
    else:
        phi_angle = phase
        lambda_angle = -phase
    gate = StandardGate.U(theta, phi_angle, lambda_angle)
    _append_conditioned_gate(
        circuit,
        gate,
        target=target,
        controls=tuple(q for q in range(num_qubits) if q != target),
        basis_pattern=mapped_reference,
    )

    for first, second in reversed(tuple(pairwise(path))):
        _append_adjacent_basis_swap(circuit, num_qubits, first, second)


def _append_adjacent_basis_swap(
    circuit: Circuit, num_qubits: int, first: int, second: int
) -> None:
    """Transpose two basis states whose bit strings differ in one position."""
    difference = first ^ second
    if difference == 0 or difference & (difference - 1):
        raise ValueError("adjacent basis states must differ in exactly one bit")
    target = difference.bit_length() - 1
    _append_conditioned_gate(
        circuit,
        StandardGate.X,
        target=target,
        controls=tuple(q for q in range(num_qubits) if q != target),
        basis_pattern=first,
    )


def _append_basis_state_phase(
    circuit: Circuit, *, num_qubits: int, basis_index: int, phase: float
) -> None:
    """Apply a phase to exactly one computational basis state."""
    target = 0
    target_is_zero = ((basis_index >> target) & 1) == 0
    if target_is_zero:
        circuit.x(target)
    _append_conditioned_gate(
        circuit,
        StandardGate.Phase(phase),
        target=target,
        controls=tuple(range(1, num_qubits)),
        basis_pattern=basis_index,
    )
    if target_is_zero:
        circuit.x(target)


def _append_conditioned_gate(
    circuit: Circuit,
    gate: StandardGate,
    *,
    target: int,
    controls: tuple[int, ...],
    basis_pattern: int,
) -> None:
    """Apply a gate when controls equal their bits in ``basis_pattern``."""
    open_controls = tuple(
        qubit for qubit in controls if ((basis_pattern >> qubit) & 1) == 0
    )
    for qubit in open_controls:
        circuit.x(qubit)
    if controls:
        circuit.append_mc_gate(gate.control(len(controls)), [*controls, target])
    else:
        circuit.append_gate(gate, [target])
    for qubit in reversed(open_controls):
        circuit.x(qubit)
