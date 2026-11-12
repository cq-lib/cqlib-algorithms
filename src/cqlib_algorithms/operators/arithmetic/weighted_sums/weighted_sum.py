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
"""WeightedSum public wrapper.

This module exposes the public :class:`WeightedSum` class of the arithmetic
package.  The wrapper accepts a list of classical weights (one per quantum
input register) together with the bit width of each register, computes the
flat per-qubit weight schedule required by the underlying synthesis helpers
and delegates circuit construction to either the QFT-based or the
computational-basis carry-chain implementation.
"""

from __future__ import annotations

from typing import Literal

from cqlib import Circuit

from ._carry_weighted_sum import build_carry_weighted_sum
from ._qft_weighted_sum import build_qft_weighted_sum

WeightedSumSynthesis = Literal["qft", "carry"]

_VALID_SYNTHESIS: frozenset[str] = frozenset({"qft", "carry"})


def _expand_weights(weights: list[int], num_inputs: list[int]) -> list[int]:
    """Return the flat per-qubit weight list.

    ``weights`` and ``num_inputs`` share the same length ``K``; for each
    input ``k`` the quantum register of width ``n_k = num_inputs[k]`` is
    interpreted as an LSB-first binary integer whose classical weight is
    ``weights[k]``.  The flat list therefore contains
    ``weights[k] * 2 ** b`` for every bit position ``b in [0, n_k)``,
    concatenated in input order.
    """

    flat: list[int] = []
    for weight, bits in zip(weights, num_inputs):
        for bit_position in range(bits):
            flat.append(weight << bit_position)
    return flat


def _compute_sum_register_width(weights: list[int], num_inputs: list[int]) -> int:
    """Return ``s``, the required sum-register bit width.

    The width is chosen so that the worst-case value (every qubit of every
    input register equals ``1``) fits into the accumulator:

    .. math::

        s = 1 + \\lfloor \\log_2( \\sum_k w_k \\cdot (2^{n_k} - 1) ) \\rfloor.
    """

    worst_case = 0
    for weight, bits in zip(weights, num_inputs):
        worst_case += weight * ((1 << bits) - 1)
    if worst_case == 0:
        return 1
    return 1 + worst_case.bit_length() - 1


class WeightedSum:
    """Multi-register quantum weighted sum with a fixed local qubit layout.

    The operator implements the unitary

    .. math::

        |x_0> |x_1> \\cdots |x_{K-1}> |0^{\\oplus s}>
            \\mapsto
        |x_0> |x_1> \\cdots |x_{K-1}>
        \\left| \\sum_{k=0}^{K-1} w_k \\cdot x_k \\right>,

    where each :math:`x_k` is an LSB-first ``num_inputs[k]``-bit quantum
    integer and :math:`w_k` is the classical integer weight associated with
    the corresponding input register.

    Parameters
        weights Classical integer weight assigned to each of the ``K``
            quantum input registers.  The ``k``-th register contributes
            ``weights[k] * value(input_k)`` to the final result.
        num_inputs Positive integer bit widths of the ``K`` input registers.
            Must have the same length as ``weights``.
        synthesis Synthesis strategy selector.  ``qft`` (the default) uses
            the zero-ancilla Ruiz-Perez 2017 QFT phase schedule; ``carry``
            uses the computational-basis multi-controlled-X carry chain
            described in the weighted-sum design document.
        approximation_degree Optional QFT approximation degree forwarded to
            the Fourier transforms when ``synthesis == 'qft'``.  Ignored for
            the ``carry`` synthesis.
        name Optional human readable circuit name.  When omitted the
            literal ``"WeightedSum"`` is used.

    Attributes
        weights Stored per-input classical weights (defensive copy).
        num_inputs Stored per-input register widths (defensive copy).
        flat_weights Per-qubit weight schedule used by the synthesis
            helpers; the entry order matches the concatenated layout of
            the input registers.
        synthesis Selected synthesis strategy.
        approximation_degree Optional QFT approximation threshold in use.
        num_state_qubits Total number of input qubits, i.e.
            ``sum(num_inputs)``.
        num_sum_qubits Width ``s`` of the LSB-first sum result register.
        num_qubits Total circuit width (state + sum + carries + helper
            when applicable).
        input_register_indices Local wire indices of each of the ``K`` input
            registers in declaration order.
        sum_register_indices Local wire indices of the LSB-first sum
            register.
        carry_register_indices Local wire indices of the intermediate
            carry qubits for the ``carry`` synthesis or ``None``.
        helper_index Local index of the dirty-helper ancilla used by the
            C3X decomposition in the ``carry`` synthesis or ``None``.
        name Human readable circuit name.
    """

    def __init__(
        self,
        weights: list[int],
        num_inputs: list[int],
        *,
        synthesis: WeightedSumSynthesis = "qft",
        approximation_degree: int | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(weights, list):
            raise TypeError(f"weights must be a list, got {weights!r}.")
        if not isinstance(num_inputs, list):
            raise TypeError(f"num_inputs must be a list, got {num_inputs!r}.")
        if len(weights) == 0:
            raise ValueError("weights must be a non-empty list.")
        if len(num_inputs) == 0:
            raise ValueError("num_inputs must be a non-empty list.")
        if len(weights) != len(num_inputs):
            raise ValueError(
                "weights and num_inputs must have the same length, got "
                f"len(weights)={len(weights)} vs len(num_inputs)="
                f"{len(num_inputs)}."
            )

        for index, entry in enumerate(weights):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(
                    "weights must be a list of integers, got "
                    f"weights[{index}]={entry!r}."
                )
        for index, entry in enumerate(weights):
            if entry < 0:
                raise ValueError(f"weights[{index}] must be >= 0, got {entry!r}.")
        for index, entry in enumerate(num_inputs):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(
                    "num_inputs must be a list of integers, got "
                    f"num_inputs[{index}]={entry!r}."
                )
        for index, entry in enumerate(num_inputs):
            if entry < 1:
                raise ValueError(f"num_inputs[{index}] must be >= 1, got {entry!r}.")

        if synthesis not in _VALID_SYNTHESIS:
            raise ValueError(
                "Unsupported WeightedSum synthesis strategy "
                f"{synthesis!r}.  Expected one of "
                f"{' '.join(sorted(_VALID_SYNTHESIS))}."
            )

        if synthesis != "qft":
            if approximation_degree is not None:
                raise ValueError(
                    "approximation_degree is only supported when synthesis='qft'."
                )
            self._approximation_degree: int | None = None
        elif approximation_degree is None:
            self._approximation_degree = None
        else:
            if isinstance(approximation_degree, bool) or not isinstance(
                approximation_degree, int
            ):
                raise TypeError(
                    "approximation_degree must be None or an integer, got "
                    f"{approximation_degree!r}."
                )
            if approximation_degree < 0:
                raise ValueError(
                    "approximation_degree must be None or a non-negative "
                    f"integer, got {approximation_degree!r}."
                )
            self._approximation_degree = int(approximation_degree)

        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {name!r}.")

        self._weights = list(weights)
        self._num_inputs = list(num_inputs)
        self._flat_weights = _expand_weights(self._weights, self._num_inputs)
        self._synthesis: WeightedSumSynthesis = synthesis
        self._name: str = name or "WeightedSum"

        n_total = sum(self._num_inputs)
        s = _compute_sum_register_width(self._weights, self._num_inputs)
        self._num_state_qubits = n_total
        self._num_sum_qubits = s

        input_indices: list[list[int]] = []
        cursor = 0
        for width in self._num_inputs:
            input_indices.append(list(range(cursor, cursor + width)))
            cursor += width
        self._input_register_indices = input_indices
        self._sum_register_indices = list(range(cursor, cursor + s))
        cursor += s

        if self._synthesis == "carry":
            num_carries = s - 1
            if num_carries > 0:
                self._carry_register_indices: list[int] | None = list(
                    range(cursor, cursor + num_carries)
                )
                cursor += num_carries
            else:
                self._carry_register_indices = None
            self._helper_index: int | None = cursor if s > 2 else None
            if self._helper_index is not None:
                cursor += 1
        else:
            self._carry_register_indices = None
            self._helper_index = None
        self._num_qubits = cursor

    # -- properties ---------------------------------------------------------

    @property
    def weights(self) -> list[int]:
        """Return the per-input classical weights."""

        return list(self._weights)

    @property
    def num_inputs(self) -> list[int]:
        """Return the per-input register widths."""

        return list(self._num_inputs)

    @property
    def flat_weights(self) -> list[int]:
        """Return the flat per-qubit weight schedule used for synthesis."""

        return list(self._flat_weights)

    @property
    def synthesis(self) -> WeightedSumSynthesis:
        """Return the selected synthesis strategy."""

        return self._synthesis

    @property
    def approximation_degree(self) -> int | None:
        """Return the optional QFT approximation threshold."""

        return self._approximation_degree

    @property
    def num_state_qubits(self) -> int:
        """Return the total number of input qubits."""

        return self._num_state_qubits

    @property
    def num_sum_qubits(self) -> int:
        """Return the width ``s`` of the sum result register."""

        return self._num_sum_qubits

    @property
    def num_qubits(self) -> int:
        """Return the total circuit width."""

        return self._num_qubits

    @property
    def input_register_indices(self) -> list[list[int]]:
        """Return the local wire indices of every input register."""

        return [list(block) for block in self._input_register_indices]

    @property
    def sum_register_indices(self) -> list[int]:
        """Return the local wire indices of the sum register."""

        return list(self._sum_register_indices)

    @property
    def carry_register_indices(self) -> list[int] | None:
        """Return the carry-wire indices for the carry synthesis."""

        if self._carry_register_indices is None:
            return None
        return list(self._carry_register_indices)

    @property
    def helper_index(self) -> int | None:
        """Return the dirty-helper index or ``None``."""

        return self._helper_index

    @property
    def name(self) -> str:
        """Return the configured human readable circuit name."""

        return self._name

    # -- circuit construction ----------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing the configured weighted sum."""

        total = self._num_qubits
        circuit = Circuit(total)

        if self._synthesis == "qft":
            sub = build_qft_weighted_sum(
                self._flat_weights,
                approximation_degree=self._approximation_degree,
            )
            qft_mapping = [
                idx for block in self._input_register_indices for idx in block
            ] + list(self._sum_register_indices)
            circuit.compose(sub, qft_mapping)
            return circuit

        sub = build_carry_weighted_sum(self._flat_weights)
        carry_mapping = [
            idx for block in self._input_register_indices for idx in block
        ] + list(self._sum_register_indices)
        if self._carry_register_indices is not None:
            carry_mapping.extend(self._carry_register_indices)
        if self._helper_index is not None:
            carry_mapping.append(self._helper_index)
        circuit.compose(sub, carry_mapping)
        return circuit
