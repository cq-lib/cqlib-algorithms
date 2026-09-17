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
"""Unified two-register adder circuit wrappers.

This module provides a single public :class:`Adder` class that exposes the
three classic two-register adder modes (half, full, modular) through a
``kind`` configuration flag.  Concrete construction is delegated to the
existing synthesis helpers in :mod:`_ripple_carry_adder` and
:mod:`_draper_qft_adder`; the wrapper is responsible for validation, a
fixed local layout and exposing the register indices consistently for
every supported mode.
"""

from __future__ import annotations

from typing import Literal, cast

from cqlib import Circuit

from ._draper_qft_adder import DraperAdderKind, build_draper_qft_adder
from ._ripple_carry_adder import build_ripple_carry_adder

AdderKind = Literal["half", "full", "modular"]
AdderSynthesis = Literal["ripple_carry", "draper_qft"]

_VALID_KINDS: frozenset[str] = frozenset({"half", "full", "modular"})
_VALID_SYNTHESES: frozenset[str] = frozenset({"ripple_carry", "draper_qft"})
_FULL_ONLY_RIPPLE_MSG = (
    "FullAdder mode only supports synthesis='ripple_carry'; "
    "no draper_qft construction is currently available."
)


class Adder:
    """Unified two-register adder with a fixed local qubit layout.

    Parameters
        num_state_qubits Positive integer width ``n`` of each input register.
        kind String adder-mode selector. Accepted values are ``half`` (carry
            out and helper, no carry in), ``full`` (carry in and carry out,
            no helper) and ``modular`` (helper, no carry wires exposed).
        synthesis String synthesis strategy name. Accepted values are
            ``ripple_carry`` for every kind and ``draper_qft`` for the half
            and modular kinds only.
        approximation_degree Optional truncation threshold forwarded to the
            draper_qft synthesis.
        name Optional human readable circuit name. When omitted a sensible
            default based on ``kind`` is chosen.

    Attributes
        kind Stored adder-mode selector.
        synthesis Stored synthesis strategy name.
        approximation_degree Optional truncation threshold for the draper_qft
            synthesis.
        num_state_qubits Stored register width ``n``.
        a_register_indices Local qubit indices of the read only ``a`` register.
            When ``kind`` is ``full`` the range is shifted one position to the
            right so that index ``0`` can host the carry in wire.
        b_register_indices Local qubit indices of the in/out register that
            holds the lower ``n`` bits of the result on output.
        carry_in_index Local index of the carry in qubit when ``kind`` is
            ``full``, otherwise ``None``.
        carry_out_index Local index of the carry out qubit for the ``half``
            and ``full`` kinds; ``None`` for modular addition.
        helper_index Local index of the scratch helper qubit present in the
            ``half`` and ``modular`` layouts; ``None`` for the full adder
            layout.
        num_qubits Total qubit count, equal to ``2n + 1`` for modular and
            ``2n + 2`` otherwise.
        name Human readable name resolved at construction time.
    """

    def __init__(
        self,
        num_state_qubits: int,
        kind: AdderKind = "half",
        synthesis: AdderSynthesis = "ripple_carry",
        *,
        approximation_degree: int | None = None,
        name: str | None = None,
    ) -> None:
        # -- num_state_qubits ------------------------------------------------
        if not isinstance(num_state_qubits, int):
            raise TypeError(
                "num_state_qubits must be an integer, got "
                f"{num_state_qubits!r}."
            )
        if num_state_qubits < 1:
            raise ValueError(
                f"num_state_qubits must be >= 1, got {num_state_qubits!r}."
            )

        # -- kind ------------------------------------------------------------
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"Unsupported Adder kind {kind!r}. "
                "Expected one of full half modular."
            )

        # -- synthesis -------------------------------------------------------
        if synthesis not in _VALID_SYNTHESES:
            raise ValueError(
                f"Unsupported Adder synthesis strategy {synthesis!r}. "
                "Expected one of draper_qft ripple_carry."
            )
        if kind == "full" and synthesis != "ripple_carry":
            raise ValueError(_FULL_ONLY_RIPPLE_MSG)

        # -- approximation_degree -------------------------------------------
        if synthesis != "draper_qft":
            if approximation_degree is not None:
                raise ValueError(
                    "approximation_degree is only supported when "
                    "synthesis='draper_qft'."
                )
            self._approximation_degree: int | None = None
        elif approximation_degree is None:
            self._approximation_degree = None
        else:
            if not isinstance(approximation_degree, int):
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

        # -- name ------------------------------------------------------------
        default_name = f"Adder[{kind},{synthesis}]"
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"name must be a string or None, got {name!r}."
            )

        # -- store & layout --------------------------------------------------
        self._num_state_qubits = int(num_state_qubits)
        self._kind = kind
        self._synthesis = synthesis
        self._name: str = name or default_name

        n = self._num_state_qubits
        if kind == "full":
            self._carry_in_index: int | None = 0
            a_start = 1
            self._helper_index: int | None = None
            self._carry_out_index: int | None = 2 * n + 1
            self._num_qubits = 2 * n + 2
        elif kind == "half":
            self._carry_in_index = None
            a_start = 0
            self._helper_index = 2 * n + 1
            self._carry_out_index = 2 * n
            self._num_qubits = 2 * n + 2
        else:  # modular
            self._carry_in_index = None
            a_start = 0
            self._helper_index = 2 * n
            self._carry_out_index = None
            self._num_qubits = 2 * n + 1

        self._a_register_indices = list(range(a_start, a_start + n))
        self._b_register_indices = list(range(a_start + n, a_start + 2 * n))

    # -- properties ---------------------------------------------------------

    @property
    def kind(self) -> str:
        """Return the stored adder-mode selector."""
        return self._kind

    @property
    def synthesis(self) -> str:
        """Return the stored synthesis strategy name."""
        return self._synthesis

    @property
    def approximation_degree(self) -> int | None:
        """Return the draper_qft approximation threshold in use."""
        return self._approximation_degree

    @property
    def num_state_qubits(self) -> int:
        """Return the width ``n`` of each of the two operand registers."""
        return self._num_state_qubits

    @property
    def a_register_indices(self) -> list[int]:
        """Return local wire indices of the read only ``a`` register."""
        return list(self._a_register_indices)

    @property
    def b_register_indices(self) -> list[int]:
        """Return local wire indices of the in/out ``b`` register."""
        return list(self._b_register_indices)

    @property
    def carry_in_index(self) -> int | None:
        """Return the local carry in qubit index or ``None``."""
        return self._carry_in_index

    @property
    def carry_out_index(self) -> int | None:
        """Return the local carry out qubit index or ``None``."""
        return self._carry_out_index

    @property
    def helper_index(self) -> int | None:
        """Return the local helper scratch qubit index or ``None``."""
        return self._helper_index

    @property
    def num_qubits(self) -> int:
        """Return the total circuit width for the configured layout."""
        return self._num_qubits

    @property
    def name(self) -> str:
        """Return the configured human-readable circuit name."""
        return self._name

    # -- circuit construction ----------------------------------------------

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing the configured adder."""
        n = self._num_state_qubits
        total = self._num_qubits
        circuit = Circuit(total)

        if self._synthesis == "ripple_carry":
            sub = build_ripple_carry_adder(n, kind=self._kind)
            circuit.compose(sub, list(range(total)))
            return circuit

        draper_circuit = build_draper_qft_adder(
            n, kind=cast(DraperAdderKind, self._kind)
        )
        if self._kind == "half":
            assert self._carry_out_index is not None
            mapping = (
                list(self._a_register_indices)
                + list(self._b_register_indices)
                + [self._carry_out_index]
            )
        else:  # modular
            mapping = (
                list(self._a_register_indices)
                + list(self._b_register_indices)
            )
        circuit.compose(draper_circuit, mapping)
        return circuit
