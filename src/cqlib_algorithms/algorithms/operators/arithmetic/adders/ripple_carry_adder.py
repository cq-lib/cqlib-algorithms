"""Ripple carry adder circuits.

This module implements the concrete ripple carry adders used by the arithmetic
package. It provides the Cuccaro MAJ and UMA primitives together with the
half, full, and modular schedules used by the public wrapper classes. The
circuits follow the shared LSB first register convention of this package.

References
Steven A Cuccaro, Thomas G Draper, Samuel A Kutin, David Petrie Moulton,
A New Quantum Ripple Carry Addition Circuit, 2004, arXiv quant ph 0410184.

Vlatko Vedral, Adriano Barenco, Artur Ekert,
Quantum Networks for Elementary Arithmetic Operations, 1996,
arXiv quant ph 9511018.
"""

from __future__ import annotations

from typing import Literal

from cqlib import Circuit

AdderKind = Literal["half", "full", "modular"]

_VALID_KINDS: frozenset[str] = frozenset({"half", "full", "modular"})


def _append_maj(
    circuit: Circuit,
    a_index: int,
    b_index: int,
    c_index: int,
) -> None:
    """Append the Cuccaro MAJ primitive.

    The exact gate order follows the paper and the OpenQASM reference.
    """
    circuit.cx(c_index, b_index)
    circuit.cx(c_index, a_index)
    circuit.ccx(a_index, b_index, c_index)


def _append_uma(
    circuit: Circuit,
    a_index: int,
    b_index: int,
    c_index: int,
) -> None:
    """Append the Cuccaro UMA primitive.

    The exact gate order follows the paper and the OpenQASM reference.
    """
    circuit.ccx(a_index, b_index, c_index)
    circuit.cx(c_index, a_index)
    circuit.cx(a_index, b_index)


class RippleCarryAdder:
    """Ripple carry adder configuration for the CDKM implementation.

    The implementation follows the layouts described in the local technical
    notes under docs/design_md/arithmetic_docs.

    full:
        |cin> |a>_n |b>_n |cout>
            ->
        |cin> |a>_n |a+b+cin mod 2^n>_n |cout xor carry_n>

    half:
        |a>_n |b>_n |cout> |helper>
            ->
        |a>_n |a+b mod 2^n>_n |cout xor carry_n> |0>

    modular:
        |a>_n |b>_n |helper>
            ->
        |a>_n |a+b mod 2^n>_n |0>
    """

    def __init__(
        self,
        num_state_qubits: int,
        kind: AdderKind = "half",
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(num_state_qubits, int) or num_state_qubits < 1:
            raise ValueError(
                "num_state_qubits must be an integer >= 1, got "
                f"{num_state_qubits!r}."
            )
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"Unknown adder kind {kind!r}. "
                f"Expected one of {sorted(_VALID_KINDS)}."
            )

        self._num_state_qubits = int(num_state_qubits)
        self._kind = kind
        self._name = name or f"RippleCarryAdder[{kind}]"

        n = self._num_state_qubits
        if kind == "full":
            self._num_qubits = 2 * n + 2
            self._carry_in_index = 0
            self._a_register_indices = list(range(1, n + 1))
            self._b_register_indices = list(range(n + 1, 2 * n + 1))
            self._carry_out_index = 2 * n + 1
            self._helper_index = None
        elif kind == "half":
            self._num_qubits = 2 * n + 2
            self._a_register_indices = list(range(n))
            self._b_register_indices = list(range(n, 2 * n))
            self._carry_in_index = None
            self._carry_out_index = 2 * n
            self._helper_index = 2 * n + 1
        else:  # modular
            self._num_qubits = 2 * n + 1
            self._a_register_indices = list(range(n))
            self._b_register_indices = list(range(n, 2 * n))
            self._carry_in_index = None
            self._carry_out_index = None
            self._helper_index = 2 * n

    @property
    def name(self) -> str:
        """Return the configured circuit name."""
        return self._name

    @property
    def num_qubits(self) -> int:
        """Return the total local width currently reserved for the circuit."""
        return self._num_qubits

    @property
    def num_state_qubits(self) -> int:
        """Return the register width n."""
        return self._num_state_qubits

    @property
    def kind(self) -> str:
        """Return the configured adder kind."""
        return self._kind

    @property
    def a_register_indices(self) -> list[int]:
        """Return local indices of register a."""
        return list(self._a_register_indices)

    @property
    def b_register_indices(self) -> list[int]:
        """Return local indices of register b."""
        return list(self._b_register_indices)

    @property
    def carry_in_index(self) -> int | None:
        """Return local index of the carry in wire when present."""
        return self._carry_in_index

    @property
    def carry_out_index(self) -> int | None:
        """Return local index of the carry out wire when present."""
        return self._carry_out_index

    @property
    def helper_index(self) -> int | None:
        """Return local index of the helper wire when present."""
        return self._helper_index

    def build_circuit(self) -> Circuit:
        """Build the CDKM adder circuit for the selected kind.

        The three variants are wired directly from the local Cuccaro notes:

        full:
            majority(cin, b[0], a[0])
            for i in 0 .. n-2:
                majority(a[i], b[i+1], a[i+1])
            cx(a[n-1], cout)
            for i in n-2 .. 0:
                unmaj(a[i], b[i+1], a[i+1])
            unmaj(cin, b[0], a[0])

        half:
            majority(helper, b[0], a[0])
            for i in 0 .. n-2:
                majority(a[i], b[i+1], a[i+1])
            cx(a[n-1], cout)
            for i in n-2 .. 0:
                unmaj(a[i], b[i+1], a[i+1])
            unmaj(helper, b[0], a[0])

        modular:
            add the low-order n-1 bits using the half-adder pattern and use
            b[n-1] as the temporary high-bit output, then finish with
            b[n-1] ^= a[n-1].
        """
        n = self._num_state_qubits
        circuit = Circuit(self._num_qubits)
        a_indices = self._a_register_indices
        b_indices = self._b_register_indices

        if self._kind == "full":
            cin_index = self._carry_in_index
            cout_index = self._carry_out_index
            assert cin_index is not None
            assert cout_index is not None

            _append_maj(
                circuit,
                cin_index,
                b_indices[0],
                a_indices[0],
            )
            for i in range(n - 1):
                _append_maj(
                    circuit,
                    a_indices[i],
                    b_indices[i + 1],
                    a_indices[i + 1],
                )

            circuit.cx(a_indices[n - 1], cout_index)

            for i in range(n - 2, -1, -1):
                _append_uma(
                    circuit,
                    a_indices[i],
                    b_indices[i + 1],
                    a_indices[i + 1],
                )

            _append_uma(
                circuit,
                cin_index,
                b_indices[0],
                a_indices[0],
            )
            return circuit

        helper_index = self._helper_index
        assert helper_index is not None

        if self._kind == "half":
            cout_index = self._carry_out_index
            assert cout_index is not None

            _append_maj(
                circuit,
                helper_index,
                b_indices[0],
                a_indices[0],
            )
            for i in range(n - 1):
                _append_maj(
                    circuit,
                    a_indices[i],
                    b_indices[i + 1],
                    a_indices[i + 1],
                )

            circuit.cx(a_indices[n - 1], cout_index)

            for i in range(n - 2, -1, -1):
                _append_uma(
                    circuit,
                    a_indices[i],
                    b_indices[i + 1],
                    a_indices[i + 1],
                )

            _append_uma(
                circuit,
                helper_index,
                b_indices[0],
                a_indices[0],
            )
            return circuit

        if n == 1:
            circuit.cx(a_indices[0], b_indices[0])
            return circuit

        _append_maj(
            circuit,
            helper_index,
            b_indices[0],
            a_indices[0],
        )
        for i in range(n - 2):
            _append_maj(
                circuit,
                a_indices[i],
                b_indices[i + 1],
                a_indices[i + 1],
            )

        circuit.cx(a_indices[n - 2], b_indices[n - 1])

        for i in range(n - 3, -1, -1):
            _append_uma(
                circuit,
                a_indices[i],
                b_indices[i + 1],
                a_indices[i + 1],
            )

        _append_uma(
            circuit,
            helper_index,
            b_indices[0],
            a_indices[0],
        )
        circuit.cx(a_indices[n - 1], b_indices[n - 1])
        return circuit


__all__ = [
    "RippleCarryAdder",
    "AdderKind",
    "_append_maj",
    "_append_uma",
]

