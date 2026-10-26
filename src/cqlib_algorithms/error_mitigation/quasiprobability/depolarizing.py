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

r"""Quasi-probability decompositions of ideal operations.

A quasi-probability decomposition rewrites one ideal operation as a signed
linear combination of implementable operations,

.. math::

    \mathcal{U} = \sum_\alpha \eta_\alpha \,
        \mathcal{D} \circ \mathcal{P}_\alpha \circ \mathcal{U},

where :math:`\mathcal{D}` is the device noise channel and
:math:`\mathcal{P}_\alpha` is a Pauli correction appended after the gate.
The noise channel is a premise, never an object: only the Pauli corrections
are executable gates, so every variant circuit this layer describes can be
run on hardware. The returned data is therefore

* the ideal instruction,
* the Pauli corrections that must be appended after it, and
* the signed weights :math:`\eta_\alpha`, whose 1-norm
  :math:`\Gamma = \sum_\alpha |\eta_\alpha|` sets the sampling overhead.

Two entry points produce the same numbers from different premises. The
closed-form route reads a single scalar noise level and never touches a
channel object; the matrix route reads a real channel, converts it to
Kraus operators, and solves for the weights. The two agree exactly on
depolarizing noise, which makes the pair a built-in cross-check.

References
----------
[1] K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum
    circuits," Physical Review Letters 119, 180509, 2017. arXiv:1612.02058.
[2] S. Endo, S. C. Benjamin, and Y. Li, "Practical quantum error mitigation for
    near-future applications," Physical Review X 8, 031027, 2018. arXiv:1712.09271.
[3] R. Takagi, "Optimal resource cost for error mitigation," Physical Review Research 3,
    033178, 2021. arXiv:2006.12509.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqlib.circuit import (
    Circuit,
    CircuitGate,
    FrozenCircuit,
    Instruction,
    StandardGate,
    ValueOperation,
)
from cqlib.device import (
    NoiseModel,
    OperationKey,
    SingleQubitNoise,
    TwoQubitNoise,
)

from .._circuits import is_measurement

_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


@dataclass(frozen=True, slots=True)
class QuasiprobabilityRepresentation:
    r"""Quasi-probability expansion of one ideal operation.

    Args:
        ideal: The ideal instruction being decomposed.
        basis: The Pauli corrections to append after ``ideal``. ``basis[0]`` is the
            identity, so the first term is the ideal operation itself.
        coefficients: The signed weights :math:`\eta_\alpha`, aligned with ``basis``.

    Raises:
        ValueError: If ``basis`` and ``coefficients`` have different lengths.

    """

    ideal: Instruction
    basis: tuple[Instruction, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject a basis and a weight list that cannot be paired up."""
        if len(self.basis) != len(self.coefficients):
            raise ValueError(
                f"basis has {len(self.basis)} corrections but coefficients "
                f"has {len(self.coefficients)} weights; they must be equal."
            )

    @property
    def norm(self) -> float:
        r"""Return the quasi-probability 1-norm :math:`\Gamma = \sum|\eta|`."""
        return float(np.sum(np.abs(self.coefficients)))


def quasiprobability_of_operation(
    instruction: Instruction,
    noise: SingleQubitNoise | TwoQubitNoise,
    /,
) -> QuasiprobabilityRepresentation:
    r"""Expand one instruction against an explicit noise channel.

    Args:
        instruction: Ideal storage instruction. Parametric gates must already carry
            their bound parameters, since a storage instruction owns none.
        noise: Noise channel acting on the same number of qubits as ``instruction``.

    Returns:
        Expansion whose basis is the identity followed by the one- and
            two-qubit Pauli corrections, in Little-Endian qubit order.

    Raises:
        TypeError: If ``noise`` is neither a :class:`SingleQubitNoise` nor a
            :class:`TwoQubitNoise`.
        ValueError: If ``instruction`` is not a standard gate with a matrix, or if the
            channel and the instruction act on different numbers of qubits.
    """
    # The channel enters only through its Kraus operators. Each Pauli
    # correction gives one implementable basis channel with Kraus operators
    # {E_i P_alpha U}; flattening those into a design matrix and solving
    # against the ideal channel yields the weights. The system is exactly
    # solvable, so the least-squares solution is the decomposition itself.
    if isinstance(noise, SingleQubitNoise):
        num_qubits = 1
    elif isinstance(noise, TwoQubitNoise):
        num_qubits = 2
    else:
        raise TypeError(
            "noise must be a cqlib SingleQubitNoise or TwoQubitNoise but was "
            f"{type(noise).__name__}."
        )

    ideal = _instruction_matrix(instruction)
    dimension = 1 << num_qubits
    if ideal.shape != (dimension, dimension):
        raise ValueError(
            f"the noise channel acts on {num_qubits} qubits but the "
            f"instruction matrix has shape {ideal.shape!r}."
        )

    kraus = [np.asarray(op, dtype=complex) for op in noise.to_kraus()]
    paulis = [_pauli_matrix(factors) for factors in _pauli_strings(num_qubits)]
    coefficients = _least_squares_coefficients(ideal, kraus, paulis)
    return QuasiprobabilityRepresentation(
        instruction,
        _pauli_basis_instructions(num_qubits),
        tuple(coefficients),
    )


def quasiprobability_of_circuit(
    circuit: Circuit,
    noise_model: NoiseModel,
    /,
) -> list[QuasiprobabilityRepresentation]:
    r"""Expand every instruction against a ``NoiseModel``.

    Args:
        circuit: Circuit whose operations are expanded.
        noise_model: Noise model holding a channel for every single- and two-qubit gate.
            Measurements are skipped, since noise is attached to gates only.

    Returns:
        One expansion per non-measurement operation, in circuit order.

    Raises:
        ValueError: If an operation is not a standard gate with an explicit gate
            definition, acts on more than two qubits, or has no channel
            registered in ``noise_model``.
    """
    # A channel is looked up by gate definition and qubit indices. Noise is
    # keyed by the unparameterised gate, so a parametric gate is matched by its
    # definition with the parameters zeroed, and a lookup that returns several
    # channels contributes its first registered one.
    representations: list[QuasiprobabilityRepresentation] = []
    for operation in circuit.operations:
        if is_measurement(operation):
            continue
        representations.append(
            quasiprobability_of_operation(
                _storage_instruction(operation),
                _lookup_noise(operation, noise_model),
            )
        )
    return representations


def depolarizing_quasiprobability_of_operation(
    instruction: Instruction,
    noise_level: float,
    /,
) -> QuasiprobabilityRepresentation:
    r"""Closed-form depolarizing expansion from a scalar noise level alone.

    Following Temme et al. (arXiv:1612.02058) and Takagi (arXiv:2006.12509),
    the depolarizing channel :math:`\mathcal{D}(\rho) = (1 - \epsilon)\rho +
    \epsilon I / 2^k` admits the minimal 1-norm expansion

    .. math::

        \eta_0 = 1 + \frac{4^k - 1}{4^k}\frac{\epsilon}{1 - \epsilon},
        \qquad
        \eta_\alpha = -\frac{1}{4^k}\frac{\epsilon}{1 - \epsilon},

    with :math:`\epsilon = \frac{4}{3}\,\text{noise\_level}` for one qubit
    and :math:`\epsilon = \frac{16}{15}\,\text{noise\_level}` for two.

    Args:
        instruction: Ideal standard-gate instruction, acting on one or two qubits.
        noise_level: Depolarizing strength of the device, in :math:`[0, 1)`.

    Returns:
        Expansion whose basis is the identity followed by the Pauli
            corrections, in Little-Endian qubit order.

    Raises:
        ValueError: If ``instruction`` is not a standard gate, acts on more than two
            qubits, or has a noise level for which :math:`\epsilon \ge 1` and the
            closed form therefore diverges.
    """
    num_qubits = _instruction_num_qubits(instruction)
    coefficients = _depolarizing_coefficients(num_qubits, noise_level)
    return QuasiprobabilityRepresentation(
        instruction,
        _pauli_basis_instructions(num_qubits),
        tuple(coefficients),
    )


def depolarizing_quasiprobability_of_circuit(
    circuit: Circuit,
    noise_level: float,
    /,
) -> list[QuasiprobabilityRepresentation]:
    r"""Closed-form depolarizing expansion of every gate in a circuit.

    Every non-measurement operation is expanded with the same
    ``noise_level``, which is the gate-independent depolarizing model. The
    expansion is positional: the returned list follows circuit order, so it
    stays aligned with the operations themselves even when a gate repeats.

    Args:
        circuit: Circuit whose operations are expanded.
        noise_level: Depolarizing strength of the device, in :math:`[0, 1)`.

    Returns:
        One expansion per non-measurement operation, in circuit order.

    Raises:
        ValueError: If an operation is not a standard gate, acts on more than two
            qubits, or has a noise level for which the closed form diverges.
    """
    representations: list[QuasiprobabilityRepresentation] = []
    for operation in circuit.operations:
        if is_measurement(operation):
            continue
        representations.append(
            depolarizing_quasiprobability_of_operation(
                _storage_instruction(operation),
                noise_level,
            )
        )
    return representations


def _instruction_num_qubits(instruction: Instruction, /) -> int:
    """Return the qubit count of a standard-gate instruction.

    Raises:
        ValueError: If the instruction carries no standard gate definition.

    """
    gate = instruction.standard_gate
    if gate is None:
        raise ValueError(
            "a quasi-probability decomposition needs a standard-gate "
            f"instruction but got a {instruction.instruction_type!r} one."
        )
    return gate.num_qubits


def _instruction_matrix(instruction: Instruction, /) -> np.ndarray:
    """Return the unitary matrix of a standard-gate instruction.

    Raises:
        ValueError: If the instruction carries no standard gate definition.

    """
    gate = instruction.standard_gate
    if gate is None:
        raise ValueError(
            "a quasi-probability decomposition needs the matrix of a "
            f"standard-gate instruction but got a "
            f"{instruction.instruction_type!r} one."
        )
    return np.asarray(gate.matrix(), dtype=complex)


def _storage_instruction(operation: ValueOperation, /) -> Instruction:
    """Return the storage instruction behind a circuit operation.

    Raises:
        ValueError: If the operation is classical control flow, which has no gate
            definition and therefore no quasi-probability decomposition.

    """
    instruction = operation.instruction.instruction
    if instruction is None:
        raise ValueError(
            f"operation {operation.name!r} is classical control flow and "
            "cannot be expanded into Pauli corrections."
        )
    return instruction


def _lookup_noise(
    operation: ValueOperation,
    noise_model: NoiseModel,
    /,
) -> SingleQubitNoise | TwoQubitNoise:
    """Return the first channel registered for one circuit operation.

    Raises:
        ValueError: If the operation is not a standard gate on one or two qubits, or if
            the noise model registers no channel for it.

    """
    instruction = _storage_instruction(operation)
    gate = instruction.standard_gate
    if gate is None:
        raise ValueError(
            f"operation {operation.name!r} has no standard gate definition, "
            "so no noise channel can be looked up for it."
        )
    qubits = operation.qubits
    channels: list[SingleQubitNoise] | list[TwoQubitNoise] | None
    if len(qubits) == 1:
        channels = noise_model.get_single_qubit_errors(
            OperationKey.new_single(gate, qubits[0])
        )
    elif len(qubits) == 2:
        channels = noise_model.get_two_qubit_errors(
            OperationKey.new_double(gate, qubits[0], qubits[1])
        )
    else:
        raise ValueError(
            f"operation {operation.name!r} acts on {len(qubits)} qubits but "
            "only single- and two-qubit gates can be represented."
        )
    if not channels:
        raise ValueError(
            f"the noise model registers no channel for {operation.name!r} on "
            f"{[repr(qubit) for qubit in qubits]}; register one with "
            "NoiseModel.add_single_qubit_error or add_two_qubit_error."
        )
    return channels[0]


def _pauli_strings(num_qubits: int, /) -> tuple[str, ...]:
    """Return the Pauli strings that label the corrections of the basis.

    The order is the identity, then the single-qubit corrections acting on
    the first qubit, then those acting on the second, then the two-qubit
    corrections.

    Raises:
        ValueError: If ``num_qubits`` is not one or two.

    """
    if num_qubits == 1:
        return ("I", "X", "Y", "Z")
    if num_qubits == 2:
        return (
            "II",
            "XI",
            "YI",
            "ZI",
            "IX",
            "IY",
            "IZ",
            "XX",
            "XY",
            "XZ",
            "YX",
            "YY",
            "YZ",
            "ZX",
            "ZY",
            "ZZ",
        )
    raise ValueError(
        f"a quasi-probability decomposition needs one or two qubits but got "
        f"{num_qubits}; consider pre-compiling your circuit."
    )


def _pauli_matrix(factors: str, /) -> np.ndarray:
    """Return the matrix of a Pauli string in Little-Endian qubit order."""
    # The first qubit of an operation is the least significant state bit, so
    # the factor of the last qubit is the outermost Kron factor.
    matrix = np.ones((1, 1), dtype=complex)
    for name in reversed(factors):
        matrix = np.kron(matrix, _PAULI_MATRICES[name])
    return matrix


def _pauli_instruction(factors: str, /) -> Instruction:
    """Wrap a Pauli string as an instruction that can be appended to a gate."""
    # Single-qubit corrections are standard gates; two-qubit corrections have
    # no native standard gate and are frozen into a circuit gate instead.
    if len(factors) == 1:
        return Instruction.from_standard_gate(StandardGate.from_name(factors))
    circuit = Circuit(len(factors))
    for index, name in enumerate(factors):
        circuit.append_gate(StandardGate.from_name(name), [index])
    frozen = FrozenCircuit(list(circuit.qubits), list(circuit.operations))
    return Instruction.from_circuit_gate(CircuitGate(f"Pauli_{factors}", frozen))


def _pauli_basis_instructions(num_qubits: int, /) -> tuple[Instruction, ...]:
    """Return the corrections of the basis, identity first."""
    return tuple(
        _pauli_instruction(factors)
        for factors in _pauli_strings(num_qubits)
    )


def _depolarizing_epsilon(num_qubits: int, noise_level: float, /) -> float:
    r"""Return the depolarizing parameter :math:`\epsilon` of the channel.

    The channel is :math:`\mathcal{D}(\rho) = (1 - \epsilon)\rho +
    \epsilon I / 2^k`, and the reported noise level connects to it through
    :math:`\epsilon = 4\epsilon_{\text{1q}}/3` for one qubit and
    :math:`\epsilon = 16\epsilon_{\text{2q}}/15` for two.

    Raises:
        ValueError: If the noise level is not finite or negative, if the instruction
            acts on more than two qubits, or if the implied :math:`\epsilon`
            reaches one, where the closed form diverges.

    """
    if not np.isfinite(noise_level) or noise_level < 0.0:
        raise ValueError(
            f"noise_level must be finite and non-negative but was "
            f"{noise_level!r}."
        )
    if num_qubits == 1:
        epsilon = 4.0 / 3.0 * noise_level
    elif num_qubits == 2:
        epsilon = 16.0 / 15.0 * noise_level
    else:
        raise ValueError(
            f"a quasi-probability decomposition needs one or two qubits but "
            f"got {num_qubits}; consider pre-compiling your circuit."
        )
    if epsilon >= 1.0:
        raise ValueError(
            f"noise_level={noise_level!r} gives the depolarizing parameter "
            f"epsilon={epsilon!r} >= 1, where the closed-form decomposition "
            "of Temme et al. (arXiv:1612.02058) diverges."
        )
    return epsilon


def _depolarizing_coefficients(
    num_qubits: int,
    noise_level: float,
    /,
) -> list[float]:
    """Return the closed-form weights of the depolarizing decomposition."""
    epsilon = _depolarizing_epsilon(num_qubits, noise_level)
    ratio = epsilon / (1.0 - epsilon)
    if num_qubits == 1:
        return [1.0 + 0.75 * ratio] + 3 * [-0.25 * ratio]
    return [1.0 + (15.0 / 16.0) * ratio] + 15 * [-(1.0 / 16.0) * ratio]


def _least_squares_coefficients(
    ideal: np.ndarray,
    kraus: list[np.ndarray],
    paulis: list[np.ndarray],
    /,
) -> list[float]:
    r"""Solve :math:`\sum_\alpha \eta_\alpha \mathcal{O}_\alpha = \mathcal{U}`.

    The ideal channel lies in the span of the basis channels, so the system
    is exactly solvable and the returned weights are the decomposition
    itself.
    """
    # Channels are flattened in row-major order. With K_i the Kraus operators
    # of a basis channel, its flattened superoperator is sum_i K_i (x) K_i*;
    # the ideal channel contributes the right-hand side U (x) U*. The design
    # matrix is wide, so the weights come from the least-squares solve.
    columns = []
    for pauli in paulis:
        operators = [operator @ pauli @ ideal for operator in kraus]
        # Each Kraus term contributes K_i (x) K_i*, an (n^2, n^2) block with n
        # the channel dimension, so the accumulation starts from an explicit
        # zero block of that shape.
        superoperator = np.zeros((ideal.size, ideal.size), dtype=complex)
        for operator in operators:
            superoperator += np.kron(operator, operator.conj())
        columns.append(superoperator.reshape(-1))
    design = np.asarray(columns).T
    target = np.kron(ideal, ideal.conj()).reshape(-1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return [float(value.real) for value in solution]
