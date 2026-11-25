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

"""Superdense coding of two classical bits with one transmitted qubit.

References
----------
[1] C. H. Bennett and S. J. Wiesner, "Communication via one- and
    two-particle operators on Einstein-Podolsky-Rosen states," Physical
    Review Letters, 69, 2881-2884, 1992.
    DOI: 10.1103/PhysRevLett.69.2881.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, final, overload

from cqlib import Circuit

from .._utils.numeric import validate_positive_integer
from ..operators.state_preparation import GHZStatePreparation
from ..protocols.sampler import Sampler, SamplerRequest, require_sampler_results

_MESSAGES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _validate_message(message: tuple[int, int]) -> None:
    if not isinstance(message, tuple) or len(message) != 2:
        raise TypeError("message must be a two-bit tuple")
    bit_flip_bit, phase_bit = message
    if type(bit_flip_bit) is not int or type(phase_bit) is not int:
        raise TypeError("message bits must be integers")
    if bit_flip_bit not in (0, 1) or phase_bit not in (0, 1):
        raise ValueError("message bits must be zero or one")


def _validate_counts(counts: tuple[int, int, int, int], shots: int) -> None:
    if not isinstance(counts, tuple) or len(counts) != 4:
        raise TypeError("decoded_counts must be a four-count tuple")
    for count in counts:
        if type(count) is not int:
            raise TypeError("decoded_counts must contain integers")
        if count < 0:
            raise ValueError("decoded_counts must be non-negative")
    if sum(counts) != shots:
        raise ValueError("decoded_counts must sum to shots")


@dataclass(frozen=True, slots=True)
class SuperdenseCodingRequest:
    """One two-bit superdense-coding request.

    ``message`` is ``(bit_flip_bit, phase_bit)``. Alice applies
    ``X**bit_flip_bit Z**phase_bit`` in operator notation; because circuit
    operations are appended in time order, the phase operation is appended
    before the bit-flip operation.

    Args:
        message: Two bits ``(bit_flip_bit, phase_bit)``.
        shots: Positive number of decoding samples.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the message or shot count is invalid.
    """

    message: tuple[int, int]
    shots: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the classical message and requested shot count."""
        _validate_message(self.message)
        validate_positive_integer(self.shots, "shots")


@dataclass(frozen=True, slots=True)
class SuperdenseCodingResult:
    """Observed decoding statistics for one sent message.

    Counts use fixed public message order ``00, 01, 10, 11``, where every
    pair is ``(bit_flip_bit, phase_bit)``. A unique empirical mode is exposed
    as ``decoded_message``; a tie returns ``None`` without inventing a
    decoder decision.

    Attributes:
        sent_message: Encoded two-bit message.
        decoded_counts: Counts in public order ``00, 01, 10, 11``.
        shots: Total decoding samples.
    """

    sent_message: tuple[int, int]
    decoded_counts: tuple[int, int, int, int]
    shots: int

    def __post_init__(self) -> None:
        _validate_message(self.sent_message)
        validate_positive_integer(self.shots, "shots")
        _validate_counts(self.decoded_counts, self.shots)

    @property
    def decoded_probabilities(self) -> tuple[float, float, float, float]:
        """Return probabilities in fixed ``00, 01, 10, 11`` order."""
        counts = self.decoded_counts
        return (
            counts[0] / self.shots,
            counts[1] / self.shots,
            counts[2] / self.shots,
            counts[3] / self.shots,
        )

    @property
    def modal_messages(self) -> tuple[tuple[int, int], ...]:
        """Return every message tied for the largest observed count."""
        maximum = max(self.decoded_counts)
        return tuple(
            message
            for message, count in zip(_MESSAGES, self.decoded_counts)
            if count == maximum
        )

    @property
    def decoded_message(self) -> tuple[int, int] | None:
        """Return the unique empirical mode, or ``None`` for a tie."""
        modes = self.modal_messages
        return modes[0] if len(modes) == 1 else None

    @property
    def correct_count(self) -> int:
        """Return the number of shots decoded as the sent message."""
        return self.decoded_counts[_MESSAGES.index(self.sent_message)]

    @property
    def success_rate(self) -> float:
        """Return the observed fraction decoded as the sent message."""
        return self.correct_count / self.shots

    @property
    def all_succeeded(self) -> bool:
        """Return whether every shot decoded as the sent message."""
        return self.correct_count == self.shots


@final
class SuperdenseCoding:
    """Encode two classical bits by transmitting one half of a Bell pair.

    Circuit position 0 is Alice's qubit and position 1 is Bob's qubit. A
    reusable two-qubit GHZ preparation creates ``|Phi+>``. Alice encodes
    ``(x, z)`` with ``X**x Z**z`` on position 0 and sends that qubit to Bob.
    Bob applies ``CX(0, 1)`` and ``H(0)`` before measuring in qubit order
    ``(0, 1)``. Under Core's requested-qubit string convention, the observed
    string ``"xz"`` directly denotes the public message ``(x, z)``.

    The complete input batch is validated and converted to Sampler requests
    before exactly one flattened Sampler call. Results report raw aggregate
    frequencies; the Sampler contract does not provide per-shot memory.

    Reference: Bennett and Wiesner, *Communication via one- and two-particle
    operators on Einstein-Podolsky-Rosen states*, Phys. Rev. Lett. 69,
    2881 (1992).

    Args:
        sampler: Sampling implementation used for Bell-basis decoding.

    Raises:
        TypeError: If the sampler or a request has the wrong type.
        ValueError: If message, circuit, or sampler-result validation fails.
    """

    __slots__ = ("_sampler",)

    def __init__(self, sampler: Sampler) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        self._sampler = sampler

    @property
    def sampler(self) -> Sampler:
        """Return the injected Sampler implementation."""
        return self._sampler

    @staticmethod
    def build_circuit(message: tuple[int, int]) -> Circuit:
        """Build an unmeasured superdense-coding circuit for ``message``."""
        _validate_message(message)
        bit_flip_bit, phase_bit = message

        circuit = Circuit(2)
        bell_pair = GHZStatePreparation(2).build_circuit("superdense_bell")
        circuit.compose(bell_pair, [0, 1])
        if phase_bit:
            circuit.z(0)
        if bit_flip_bit:
            circuit.x(0)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.validate()
        return circuit

    @overload
    def run(
        self, message: Sequence[SuperdenseCodingRequest], /
    ) -> list[SuperdenseCodingResult]: ...

    @overload
    def run(
        self, message: tuple[int, int], /, *, shots: int
    ) -> SuperdenseCodingResult: ...

    def run(
        self,
        message: tuple[int, int] | Sequence[SuperdenseCodingRequest],
        /,
        *,
        shots: int | None = None,
    ) -> SuperdenseCodingResult | list[SuperdenseCodingResult]:
        """Encode and decode one two-bit message or an ordered request batch.

        Args:
            message: Pair (bit-flip bit, phase bit), or a batch of coding requests.
            shots: Positive decoding samples for a single message.

        Returns:
            One decoding result, or results in the original batch order.

        Raises:
            TypeError: For invalid message types or mixed invocation styles.
            ValueError: For nonbinary message bits, invalid shots, or sampler results.
        """
        if shots is not None:
            if not isinstance(message, tuple):
                raise TypeError("a single message must be a two-bit tuple")
            request = SuperdenseCodingRequest(cast(tuple[int, int], message), shots)
            return self._run_batch([request])[0]
        return self._run_batch(cast(Sequence[SuperdenseCodingRequest], message))

    def _run_batch(
        self,
        requests: Sequence[SuperdenseCodingRequest],
    ) -> list[SuperdenseCodingResult]:
        """Execute a complete ordered batch with one Sampler call."""
        prepared: list[tuple[SuperdenseCodingRequest, SamplerRequest]] = []
        for request in requests:
            if not isinstance(request, SuperdenseCodingRequest):
                raise TypeError(
                    "requests must contain SuperdenseCodingRequest instances"
                )
            request.validate()
            circuit = self.build_circuit(request.message)
            prepared.append(
                (
                    request,
                    SamplerRequest(circuit, tuple(circuit.qubits), request.shots),
                )
            )

        if not prepared:
            return []

        sampler_requests = [sampler_request for _, sampler_request in prepared]
        sample_results = require_sampler_results(
            self._sampler.run(sampler_requests), sampler_requests
        )

        results: list[SuperdenseCodingResult] = []
        for (request, _), sample_result in zip(prepared, sample_results):
            decoded_counts = (
                sample_result.counts.get("00", 0),
                sample_result.counts.get("01", 0),
                sample_result.counts.get("10", 0),
                sample_result.counts.get("11", 0),
            )
            results.append(
                SuperdenseCodingResult(
                    sent_message=request.message,
                    decoded_counts=decoded_counts,
                    shots=request.shots,
                )
            )
        return results
