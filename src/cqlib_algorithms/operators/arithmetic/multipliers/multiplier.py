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
"""Multiplier wrapper.

This module exposes the public multiplier interface of the arithmetic package.
It provides a fixed local layout shared by the available multiplier synthesis
strategies and delegates circuit construction to the corresponding internal
implementations.
"""

from __future__ import annotations

from cqlib import Circuit

from cqlib_algorithms.operators.arithmetic.multipliers._draper_qft_multiplier import (
    build_draper_qft_multiplier,
)
from cqlib_algorithms.operators.arithmetic.multipliers._hrs_cumulative_multiplier import (
    build_hrs_cumulative_multiplier,
)


MultiplierSynthesis = str

_VALID_SYNTHESIS: frozenset[str] = frozenset(
    {"hrs_cumulative", "draper_qft"}
)


def _validate_num_state_qubits(num_state_qubits: int) -> int:
    """Validate and normalize the multiplier register width."""
    if not isinstance(num_state_qubits, int) or num_state_qubits < 1:
        raise ValueError(
            "num_state_qubits must be an integer >= 1, got "
            f"{num_state_qubits!r}."
        )
    return int(num_state_qubits)


def _validate_name(name: str | None) -> str | None:
    """Validate the optional human-readable circuit name."""
    if name is not None and not isinstance(name, str):
        raise ValueError(f"name must be a string or None, got {name!r}.")
    return name


def _validate_synthesis(synthesis: str) -> str:
    """Validate the selected multiplier synthesis strategy."""
    if synthesis not in _VALID_SYNTHESIS:
        raise ValueError(
            f"Unsupported Multiplier synthesis strategy {synthesis!r}. "
            f"Expected one of {' '.join(sorted(_VALID_SYNTHESIS))}."
        )
    return synthesis


def _validate_approximation_degree(
    approximation_degree: int | None,
    synthesis: str,
) -> int | None:
    """Validate the optional QFT approximation configuration."""
    if synthesis != "draper_qft":
        if approximation_degree is not None:
            raise ValueError(
                "approximation_degree is only supported when "
                "synthesis='draper_qft'."
            )
        return None

    if approximation_degree is None:
        return None
    if (
        not isinstance(approximation_degree, int)
        or approximation_degree < 0
    ):
        raise ValueError(
            "approximation_degree must be None or a non-negative "
            f"integer, got {approximation_degree!r}."
        )
    return int(approximation_degree)


class Multiplier:
    """Multiplier with a fixed local layout.

    Parameters
        num_state_qubits Positive integer width n of the two input registers.
        synthesis String synthesis strategy name. Accepted values are
            hrs_cumulative and draper_qft. Default hrs_cumulative.
        approximation_degree Optional truncation threshold used only by the
            draper_qft synthesis.
        name Optional human readable circuit name defaulting to Multiplier.

    Attributes
        num_state_qubits Stored register width n.
        synthesis Selected synthesis strategy name.
        approximation_degree Approximate QFT threshold in use for the
            draper_qft synthesis, otherwise None.
        a_register_indices Local qubit indices of the read only a register.
        b_register_indices Local qubit indices of the read only b register.
        product_register_indices Local qubit indices of the 2n-bit product
            register.
        helper_index Local qubit index of the helper wire for the
            hrs_cumulative synthesis, otherwise None.
        num_qubits Total qubit count, equal to 4n + 1 for hrs_cumulative and
            4n for draper_qft.
        name Human readable name.
    """

    def __init__(
        self,
        num_state_qubits: int,
        synthesis: str = "hrs_cumulative",
        *,
        approximation_degree: int | None = None,
        name: str | None = None,
    ) -> None:
        n = _validate_num_state_qubits(num_state_qubits)
        synthesis = _validate_synthesis(synthesis)
        self._approximation_degree = _validate_approximation_degree(
            approximation_degree,
            synthesis,
        )
        self._name = _validate_name(name) or "Multiplier"

        self._num_state_qubits = n
        self._synthesis = synthesis
        self._a_register_indices = list(range(0, n))
        self._b_register_indices = list(range(n, 2 * n))
        self._product_register_indices = list(range(2 * n, 4 * n))
        self._helper_index = 4 * n if synthesis == "hrs_cumulative" else None
        self._num_qubits = 4 * n + 1 if self._helper_index is not None else 4 * n

    @property
    def name(self) -> str:
        """Return the configured circuit name."""
        return self._name

    @property
    def num_state_qubits(self) -> int:
        """Return the width n of the input registers."""
        return self._num_state_qubits

    @property
    def num_qubits(self) -> int:
        """Return the total local circuit width."""
        return self._num_qubits

    @property
    def synthesis(self) -> MultiplierSynthesis:
        """Return the selected synthesis strategy."""
        return self._synthesis

    @property
    def approximation_degree(self) -> int | None:
        """Return the optional QFT approximation setting."""
        return self._approximation_degree

    @property
    def a_register_indices(self) -> list[int]:
        """Return local wire indices of register a."""
        return list(self._a_register_indices)

    @property
    def b_register_indices(self) -> list[int]:
        """Return local wire indices of register b."""
        return list(self._b_register_indices)

    @property
    def product_register_indices(self) -> list[int]:
        """Return local wire indices of the product register."""
        return list(self._product_register_indices)

    @property
    def helper_index(self) -> int | None:
        """Return the local index of the helper qubit when present."""
        return self._helper_index

    def build_circuit(self) -> Circuit:
        """Return a fresh Circuit implementing this multiplier instance."""
        if self._synthesis == "hrs_cumulative":
            return build_hrs_cumulative_multiplier(self._num_state_qubits)
        return build_draper_qft_multiplier(
            self._num_state_qubits,
            approximation_degree=self._approximation_degree,
        )
