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

r"""Noise-amplified quasi-probability expansions for PEA.

Probabilistic error amplification trades the sampling overhead of PEC for a
cheaper representation. Instead of inverting the device noise, it worsens it
in a controlled way: the ideal operation is expanded as a genuine probability
distribution over the ideal operation itself plus every Pauli-corrected copy,

.. math::

    \mathcal{U} = (1 - \epsilon') \mathcal{U}
        + \frac{\epsilon'}{4^k - 1}
          \sum_{\alpha \neq 0} \mathcal{D} \circ \mathcal{P}_\alpha
          \circ \mathcal{U},

so all weights are non-negative and :math:`\Gamma = 1`: sampling the expansion
costs no variance amplification, it merely injects extra noise of strength
:math:`\epsilon' = s\,\epsilon`. Unlike the signed PEC weights of
:mod:`cqlib_algorithms.error_mitigation.quasiprobability.depolarizing`, these
weights form a genuine probability distribution;
:mod:`cqlib_algorithms.error_mitigation.techniques.pea` then extrapolates the
expectation values back to :math:`s = 0`.
"""

from __future__ import annotations

import numpy as np

from cqlib.circuit import Instruction

from .depolarizing import (
    QuasiprobabilityRepresentation,
    _instruction_num_qubits,
    _pauli_basis_instructions,
)


def amplify_depolarizing(
    instruction: Instruction,
    noise_level: float,
    scale_factor: float,
    /,
) -> QuasiprobabilityRepresentation:
    r"""Expand one instruction at an amplified depolarizing noise level.

    The device is assumed to depolarize the gate with strength
    ``noise_level``; the returned expansion describes the same gate under
    the stronger noise :math:`\epsilon' = \text{scale\_factor} \times
    \text{noise\_level}` (Kim et al., Nature 618, 500 (2023)). The identity
    correction keeps weight :math:`1 - \epsilon'` and each of the
    :math:`4^k - 1` Pauli corrections keeps :math:`\epsilon' / (4^k - 1)`, so
    the weights are non-negative and sum to one.

    At ``scale_factor = 1`` the expansion is the baseline amplification: its
    identity term is the untouched circuit, which is the variant circuit the
    extrapolation starts from.

    Parameters
    ----------
    instruction
        Ideal standard-gate instruction, acting on one or two qubits.
    noise_level
        Baseline depolarizing strength of the device, in :math:`[0, 1)`.
    scale_factor
        Amplification factor :math:`s > 0` applied to ``noise_level``.

    Returns
    -------
    QuasiprobabilityRepresentation
        Expansion whose basis is the identity followed by the Pauli
        corrections, in Little-Endian qubit order, with
        :math:`\Gamma = 1`.

    Raises
    ------
    ValueError
        If ``instruction`` is not a standard gate or acts on more than two
        qubits, if ``noise_level`` is not finite or negative, if
        ``scale_factor`` is not finite or non-positive, or if the amplified
        level reaches one, where the expansion stops being a probability
        distribution.

    References
    ----------
    .. [1] Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. van den Berg,
       S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, and
       A. Kandala,
       "Evidence for the utility of quantum computing before fault
       tolerance," Nature 618, 500-505, 2023.
       doi:10.1038/s41586-023-06096-3.
    .. [2] A. Mari, N. Shammah, and W. J. Zeng,
       "Extending quantum probabilistic error cancellation by noise scaling,"
       Physical Review A 104, 052607, 2021. arXiv:2108.02237.
    """
    num_qubits = _instruction_num_qubits(instruction)
    amplified_level = _amplified_noise_level(noise_level, scale_factor)
    coefficients = _amplification_coefficients(num_qubits, amplified_level)
    return QuasiprobabilityRepresentation(
        instruction,
        _pauli_basis_instructions(num_qubits),
        tuple(coefficients),
    )


def _amplified_noise_level(
    noise_level: float,
    scale_factor: float,
    /,
) -> float:
    r"""Return the amplified noise level :math:`s\,\epsilon`.

    Raises
    ------
    ValueError
        If ``noise_level`` is not finite or negative, or if
        ``scale_factor`` is not finite or non-positive.
    """
    if not np.isfinite(noise_level) or noise_level < 0.0:
        raise ValueError(
            f"noise_level must be finite and non-negative but was "
            f"{noise_level!r}."
        )
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError(
            f"scale_factor must be finite and positive but was "
            f"{scale_factor!r}."
        )
    return scale_factor * noise_level


def _amplification_coefficients(
    num_qubits: int,
    amplified_level: float,
    /,
) -> list[float]:
    r"""Return the non-negative weights of the amplified expansion.

    For :math:`k` qubits the untouched gate keeps :math:`1 - \epsilon'` and
    each of the :math:`4^k - 1` Pauli corrections takes
    :math:`\epsilon' / (4^k - 1)`; the weights sum to one, hence
    :math:`\Gamma = 1`.

    Raises
    ------
    ValueError
        If ``num_qubits`` is not one or two, or if ``amplified_level``
        reaches one, where the identity weight would no longer be
        non-negative and the expansion would stop being a probability
        distribution.
    """
    if num_qubits == 1:
        corrections = 3
    elif num_qubits == 2:
        corrections = 15
    else:
        raise ValueError(
            f"PEA can only amplify single- and two-qubit gates but got "
            f"{num_qubits}; consider pre-compiling your circuit."
        )
    if amplified_level >= 1.0:
        raise ValueError(
            f"the amplified depolarizing level {amplified_level!r} reaches "
            "one, beyond which the amplification weights would become "
            "negative; PEA needs an all-non-negative expansion with "
            "Gamma = 1."
        )
    correction_weight = amplified_level / corrections
    return [1.0 - amplified_level] + corrections * [correction_weight]
