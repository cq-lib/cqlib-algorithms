r"""Projector-controlled phase rotations for standard QSVT."""

from __future__ import annotations

import math
from numbers import Real
from typing import final

from cqlib.circuit.gates import StandardGate

from cqlib import Circuit


@final
class ProjectorControlledPhase:
    r"""Build ``exp(i phase (2 \Pi - I))`` for a standard zero-projector ``\Pi``.

    The local qubit layout is always ``[system..., ancilla...]``. For a
    standard block encoding, the projector is

    .. math::

        \Pi = |0\rangle\langle 0|^{\otimes a} \otimes I,

    so the phase rotation multiplies the all-zero ancilla subspace by
    ``exp(i phase)`` and the orthogonal complement by ``exp(-i phase)``.

    Parameters
    ----------
    num_system_qubits
        Number of system qubits.
    num_ancilla_qubits
        Number of ancilla qubits defining the standard projector.
    phase
        Real phase angle in radians.

    Raises
    ------
    TypeError
        If any argument has an invalid type.
    ValueError
        If any qubit count is negative or the total width is zero.
    """

    __slots__ = ("_num_ancilla_qubits", "_num_system_qubits", "_phase")

    _num_system_qubits: int
    _num_ancilla_qubits: int
    _phase: float

    def __init__(
        self,
        num_system_qubits: int,
        num_ancilla_qubits: int,
        *,
        phase: float,
    ) -> None:
        self._num_system_qubits = _validate_non_negative_integer(
            num_system_qubits,
            name="num_system_qubits",
        )
        self._num_ancilla_qubits = _validate_non_negative_integer(
            num_ancilla_qubits,
            name="num_ancilla_qubits",
        )
        if self._num_system_qubits + self._num_ancilla_qubits <= 0:
            raise ValueError("total qubit count must be greater than zero")
        self._phase = _validate_real_phase(phase)

    @property
    def phase(self) -> float:
        r"""Return the configured phase angle."""

        return self._phase

    @property
    def num_system_qubits(self) -> int:
        r"""Return the system-register width."""

        return self._num_system_qubits

    @property
    def num_ancilla_qubits(self) -> int:
        r"""Return the ancilla-register width."""

        return self._num_ancilla_qubits

    @property
    def num_qubits(self) -> int:
        r"""Return the total local circuit width."""

        return self.num_system_qubits + self.num_ancilla_qubits

    @property
    def system_register_indices(self) -> list[int]:
        r"""Return the local system-register indices."""

        return list(range(self.num_system_qubits))

    @property
    def ancilla_register_indices(self) -> list[int]:
        r"""Return the local ancilla-register indices."""

        return list(range(self.num_system_qubits, self.num_qubits))

    def with_phase(self, phase: float) -> ProjectorControlledPhase:
        r"""Return a new projector-phase operator with the same registers."""

        return ProjectorControlledPhase(
            self.num_system_qubits,
            self.num_ancilla_qubits,
            phase=phase,
        )

    def build_circuit(self) -> Circuit:
        r"""Build and return a fresh local-numbering projector-phase circuit."""

        circuit = Circuit(self.num_qubits)
        circuit.set_global_phase(-self.phase)
        ancillas = self.ancilla_register_indices

        if not ancillas:
            circuit.set_global_phase(self.phase)
            return circuit

        for qubit in ancillas:
            circuit.x(qubit)

        if len(ancillas) == 1:
            circuit.append_gate(StandardGate.Phase(2.0 * self.phase), [ancillas[0]])
        else:
            target = ancillas[-1]
            controls = ancillas[:-1]
            controlled_phase = StandardGate.Phase(2.0 * self.phase).control(
                len(controls)
            )
            circuit.append_mc_gate(controlled_phase, [*controls, target])

        for qubit in range(len(ancillas) - 1, -1, -1):
            circuit.x(ancillas[qubit])
        return circuit


def _validate_non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _validate_real_phase(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("phase must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TypeError("phase must be a real finite scalar")
    return scalar
