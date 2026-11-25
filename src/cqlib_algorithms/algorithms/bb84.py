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

"""Prepare-and-measure BB84 protocol experiments.

References
----------
[1] C. H. Bennett and G. Brassard, "Quantum cryptography: Public key
    distribution and coin tossing," Theoretical Computer Science, 560,
    7-11, 2014.
    DOI: 10.1016/j.tcs.2014.05.025.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from numbers import Real
from typing import cast, overload

from cqlib import Circuit

from ..operators.state_preparation import BasisStateInitialization
from ..protocols.sampler import Sampler, SamplerRequest, require_sampler_results


class BB84Basis(Enum):
    """Identify the computational (Z) and diagonal (X) BB84 bases."""

    Z = "Z"
    X = "X"


def _copy_bits(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of bits")
    copied = tuple(values)
    _validate_bits(copied, name)
    return copied


def _validate_bits(values: tuple[int, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of bits")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain only integer bits")
    if any(value not in (0, 1) for value in values):
        raise ValueError(f"{name} must contain only 0 or 1")


def _copy_bases(
    values: Sequence[BB84Basis],
    name: str,
) -> tuple[BB84Basis, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of BB84Basis values")
    copied = tuple(values)
    _validate_bases(copied, name)
    return copied


def _validate_bases(values: tuple[BB84Basis, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of BB84Basis values")
    if any(not isinstance(value, BB84Basis) for value in values):
        raise TypeError(f"{name} must contain only BB84Basis values")


def _copy_intercept_bases(
    values: Sequence[BB84Basis | None],
) -> tuple[BB84Basis | None, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            "intercept_resend_bases must be a sequence of BB84Basis or None values"
        )
    copied = tuple(values)
    _validate_intercept_bases(copied)
    return copied


def _validate_intercept_bases(
    values: tuple[BB84Basis | None, ...],
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(
            "intercept_resend_bases must be a tuple of BB84Basis or None values"
        )
    if any(value is not None and not isinstance(value, BB84Basis) for value in values):
        raise TypeError(
            "intercept_resend_bases must contain only BB84Basis or None values"
        )


def _copy_indices(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    copied = tuple(values)
    _validate_indices(copied, name)
    return copied


def _validate_indices(values: tuple[int, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of integers")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain only integers")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must be non-negative")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _validate_max_qber(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("max_qber must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("max_qber must be finite")
    if not 0.0 <= numeric <= 1.0:
        raise ValueError("max_qber must be between zero and one")


@dataclass(frozen=True, slots=True)
class BB84Request:
    """One explicitly randomized BB84 prepare-and-measure experiment.

    All bits and bases are supplied by the caller. This component deliberately
    does not create cryptographic or pseudo-random choices. Parameter-estimation
    indices refer to original signal positions, must be sorted and unique, and
    may select only positions where Alice and Bob chose the same basis.

    ``intercept_resend_bases`` enables the textbook independent intercept-resend
    experiment. Each non-``None`` entry makes Eve measure that signal in the
    supplied basis and resend the state represented by the actual outcome;
    ``None`` leaves that signal untouched. It is not a model of a general
    attack.

    Args:
        alice_bits: Alice's prepared bits in transmission order.
        alice_bases: Alice's basis for each signal.
        bob_bases: Bob's measurement basis for each signal.
        max_qber: Accepted sampled quantum bit-error rate.
        parameter_estimation_indices: Sifted positions disclosed for testing.
        intercept_resend_bases: Optional Eve basis per signal.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If lengths, indices, bits, bases, or QBER are invalid.
    """

    alice_bits: tuple[int, ...]
    alice_bases: tuple[BB84Basis, ...]
    bob_bases: tuple[BB84Basis, ...]
    _: KW_ONLY
    max_qber: float
    parameter_estimation_indices: tuple[int, ...] = ()
    intercept_resend_bases: tuple[BB84Basis | None, ...] | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "alice_bits", _copy_bits(self.alice_bits, "alice_bits")
        )
        object.__setattr__(
            self,
            "alice_bases",
            _copy_bases(self.alice_bases, "alice_bases"),
        )
        object.__setattr__(
            self,
            "bob_bases",
            _copy_bases(self.bob_bases, "bob_bases"),
        )
        object.__setattr__(
            self,
            "parameter_estimation_indices",
            _copy_indices(
                self.parameter_estimation_indices,
                "parameter_estimation_indices",
            ),
        )
        _validate_max_qber(self.max_qber)
        object.__setattr__(self, "max_qber", float(self.max_qber))
        if self.intercept_resend_bases is not None:
            object.__setattr__(
                self,
                "intercept_resend_bases",
                _copy_intercept_bases(self.intercept_resend_bases),
            )
        self._validate_relationships()

    def validate(self) -> None:
        """Revalidate the immutable request fields before execution."""
        _validate_bits(self.alice_bits, "alice_bits")
        _validate_bases(self.alice_bases, "alice_bases")
        _validate_bases(self.bob_bases, "bob_bases")
        _validate_indices(
            self.parameter_estimation_indices,
            "parameter_estimation_indices",
        )
        _validate_max_qber(self.max_qber)
        if self.intercept_resend_bases is not None:
            _validate_intercept_bases(self.intercept_resend_bases)
        self._validate_relationships()

    def _validate_relationships(self) -> None:
        if not self.alice_bits:
            raise ValueError("alice_bits must not be empty")
        if len(self.alice_bases) != len(self.alice_bits):
            raise ValueError("alice_bases must have one entry per Alice bit")
        if len(self.bob_bases) != len(self.alice_bits):
            raise ValueError("bob_bases must have one entry per Alice bit")
        if self.intercept_resend_bases is not None and len(
            self.intercept_resend_bases
        ) != len(self.alice_bits):
            raise ValueError("intercept_resend_bases must have one entry per Alice bit")

        for index in self.parameter_estimation_indices:
            if index >= len(self.alice_bits):
                raise ValueError("parameter_estimation_indices must be in range")
            if self.alice_bases[index] is not self.bob_bases[index]:
                raise ValueError(
                    "parameter_estimation_indices must select matching bases"
                )


@dataclass(frozen=True, slots=True)
class BB84Result:
    """Sifting and parameter-estimation data from one BB84 experiment.

    The remaining bits are only *candidate raw-key material*. The result does
    not claim secrecy and does not perform authentication, information
    reconciliation, privacy amplification, or finite-key security analysis.

    Attributes:
        sifted_indices: Positions where Alice and Bob used the same basis.
        parameter_estimation_indices: Disclosed positions used for QBER.
        alice_sifted_bits: Alice's bits at sifted positions.
        bob_sifted_bits: Bob's bits at sifted positions.
        max_qber: Acceptance threshold copied from the request.
    """

    sifted_indices: tuple[int, ...]
    parameter_estimation_indices: tuple[int, ...]
    alice_sifted_bits: tuple[int, ...]
    bob_sifted_bits: tuple[int, ...]
    max_qber: float

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        sifted_indices = _copy_indices(self.sifted_indices, "sifted_indices")
        parameter_indices = _copy_indices(
            self.parameter_estimation_indices,
            "parameter_estimation_indices",
        )
        alice_bits = _copy_bits(self.alice_sifted_bits, "alice_sifted_bits")
        bob_bits = _copy_bits(self.bob_sifted_bits, "bob_sifted_bits")
        _validate_max_qber(self.max_qber)
        if len(alice_bits) != len(sifted_indices):
            raise ValueError("alice_sifted_bits must align with sifted_indices")
        if len(bob_bits) != len(sifted_indices):
            raise ValueError("bob_sifted_bits must align with sifted_indices")
        sifted_set = set(sifted_indices)
        if any(index not in sifted_set for index in parameter_indices):
            raise ValueError("parameter_estimation_indices must be sifted indices")
        object.__setattr__(self, "sifted_indices", sifted_indices)
        object.__setattr__(self, "parameter_estimation_indices", parameter_indices)
        object.__setattr__(self, "alice_sifted_bits", alice_bits)
        object.__setattr__(self, "bob_sifted_bits", bob_bits)
        object.__setattr__(self, "max_qber", float(self.max_qber))

    @property
    def sifted_count(self) -> int:
        """Return the number of positions retained after basis sifting."""
        return len(self.sifted_indices)

    @property
    def parameter_estimation_count(self) -> int:
        """Return the number of publicly compared sifted positions."""
        return len(self.parameter_estimation_indices)

    @property
    def parameter_estimation_error_count(self) -> int:
        """Return the number of disagreements in the disclosed sample."""
        positions = self._sifted_positions_by_index()
        return sum(
            self.alice_sifted_bits[positions[index]]
            != self.bob_sifted_bits[positions[index]]
            for index in self.parameter_estimation_indices
        )

    @property
    def qber(self) -> float | None:
        """Return sampled QBER, or ``None`` when no positions were disclosed."""
        if not self.parameter_estimation_indices:
            return None
        return self.parameter_estimation_error_count / len(
            self.parameter_estimation_indices
        )

    @property
    def parameter_estimation_passed(self) -> bool | None:
        """Return the sampled threshold verdict, if a sample was supplied."""
        qber = self.qber
        return None if qber is None else qber <= self.max_qber

    @property
    def candidate_indices(self) -> tuple[int, ...]:
        """Return sifted positions not disclosed for parameter estimation."""
        disclosed = set(self.parameter_estimation_indices)
        return tuple(index for index in self.sifted_indices if index not in disclosed)

    @property
    def alice_candidate_bits(self) -> tuple[int, ...]:
        """Return Alice's undisclosed candidate raw-key bits."""
        disclosed = set(self.parameter_estimation_indices)
        return tuple(
            bit
            for index, bit in zip(self.sifted_indices, self.alice_sifted_bits)
            if index not in disclosed
        )

    @property
    def bob_candidate_bits(self) -> tuple[int, ...]:
        """Return Bob's undisclosed candidate raw-key bits."""
        disclosed = set(self.parameter_estimation_indices)
        return tuple(
            bit
            for index, bit in zip(self.sifted_indices, self.bob_sifted_bits)
            if index not in disclosed
        )

    @property
    def candidate_count(self) -> int:
        """Return the number of undisclosed sifted positions."""
        return len(self.candidate_indices)

    @property
    def candidate_keys_match(self) -> bool:
        """Return whether Alice and Bob have equal candidate raw-key bits."""
        return self.alice_candidate_bits == self.bob_candidate_bits

    def _sifted_positions_by_index(self) -> dict[int, int]:
        return {index: position for position, index in enumerate(self.sifted_indices)}


def _build_prepare_measure_circuit(
    bit: int,
    preparation_basis: BB84Basis,
    measurement_basis: BB84Basis,
) -> Circuit:
    circuit = BasisStateInitialization(1, bit).build_circuit("bb84")
    if preparation_basis is BB84Basis.X:
        circuit.h(0)
    if measurement_basis is BB84Basis.X:
        circuit.h(0)
    return circuit


def _sample_request(circuit: Circuit) -> SamplerRequest:
    return SamplerRequest(circuit, tuple(circuit.qubits), 1)


def _observed_bit(result_counts: dict[str, int]) -> int:
    observed = [bitstring for bitstring, count in result_counts.items() if count == 1]
    if len(observed) != 1:
        raise ValueError("one-shot sampler result must contain one observed outcome")
    return int(observed[0])


class BB84:
    """Run explicitly configured BB84 prepare-and-measure experiments.

    Without interception, every signal is prepared by Alice and measured by
    Bob using one flattened Sampler call. With ``intercept_resend_bases``, a
    first flattened call measures all intercepted signals; the observed Eve
    bits, rather than Alice's original bits, then determine states in a second
    flattened Bob call. ``None`` entries bypass Eve. Every signal uses one shot.

    The class models teaching and protocol experiments only. It does not
    provide authentication, error correction, privacy amplification,
    composable finite-key security, device characterization, or protection
    against general attacks. Its undisclosed output is candidate raw-key
    material, not a production cryptographic key.

    References: Bennett and Brassard, *Quantum cryptography: Public key
    distribution and coin tossing* (1984); Tomamichel et al., *Tight finite-key
    analysis for quantum cryptography*, Nature Communications 3, 634 (2012).

    Args:
        sampler: Sampling implementation used for signal measurements.

    Raises:
        TypeError: If the sampler or a request has the wrong type.
        ValueError: If protocol inputs or sampler results are inconsistent.
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

    @overload
    def run(self, alice_bits: Sequence[BB84Request], /) -> list[BB84Result]: ...

    @overload
    def run(
        self,
        alice_bits: tuple[int, ...],
        alice_bases: tuple[BB84Basis, ...],
        bob_bases: tuple[BB84Basis, ...],
        *,
        max_qber: float,
        parameter_estimation_indices: tuple[int, ...] = (),
        intercept_resend_bases: tuple[BB84Basis | None, ...] | None = None,
    ) -> BB84Result: ...

    def run(
        self,
        alice_bits: tuple[int, ...] | Sequence[BB84Request],
        alice_bases: tuple[BB84Basis, ...] | None = None,
        bob_bases: tuple[BB84Basis, ...] | None = None,
        *,
        max_qber: float | None = None,
        parameter_estimation_indices: tuple[int, ...] | None = None,
        intercept_resend_bases: tuple[BB84Basis | None, ...] | None = None,
    ) -> BB84Result | list[BB84Result]:
        """Prepare, measure, and sift one signal sequence or an ordered request batch.

        Args:
            alice_bits: Prepared bits in transmission order, or a request batch.
            alice_bases: Preparation basis of each signal for a single task.
            bob_bases: Measurement basis of each signal for a single task.
            max_qber: Accepted disclosed-sample error rate for a single task.
            parameter_estimation_indices: Original matching-basis positions disclosed for testing.
            intercept_resend_bases: Optional Eve measurement basis per signal.

        Returns:
            Sifted bits and disclosed-sample diagnostics for one task, or
            results in the original batch order.

        Raises:
            TypeError: For invalid types or missing single-task settings.
            ValueError: For inconsistent bits, bases, disclosure indices, or QBER.
        """
        if any(
            value is not None
            for value in (
                alice_bases,
                bob_bases,
                max_qber,
                parameter_estimation_indices,
                intercept_resend_bases,
            )
        ):
            if alice_bases is None or bob_bases is None or max_qber is None:
                raise TypeError(
                    "single BB84 requires alice_bases, bob_bases, and max_qber"
                )
            if not isinstance(alice_bits, tuple):
                raise TypeError("alice_bits must be a tuple for a single task")
            request = BB84Request(
                cast(tuple[int, ...], alice_bits),
                alice_bases,
                bob_bases,
                max_qber=max_qber,
                parameter_estimation_indices=()
                if parameter_estimation_indices is None
                else parameter_estimation_indices,
                intercept_resend_bases=intercept_resend_bases,
            )
            return self._run_batch([request])[0]
        return self._run_batch(cast(Sequence[BB84Request], alice_bits))

    def _run_batch(self, requests: Sequence[BB84Request]) -> list[BB84Result]:
        """Execute a request batch and return sifting results in input order."""
        validated: list[BB84Request] = []
        alice_circuits: list[list[Circuit]] = []
        eve_sampler_requests: list[SamplerRequest] = []
        eve_locations: list[tuple[int, int]] = []

        for request_index, request in enumerate(requests):
            if not isinstance(request, BB84Request):
                raise TypeError("requests must contain BB84Request instances")
            request.validate()
            validated.append(request)
            request_circuits: list[Circuit] = []
            for signal_index, (bit, alice_basis) in enumerate(
                zip(request.alice_bits, request.alice_bases)
            ):
                circuit = _build_prepare_measure_circuit(
                    bit,
                    alice_basis,
                    request.bob_bases[signal_index],
                )
                request_circuits.append(circuit)
                if request.intercept_resend_bases is not None:
                    eve_basis = request.intercept_resend_bases[signal_index]
                    if eve_basis is None:
                        continue
                    eve_circuit = _build_prepare_measure_circuit(
                        bit,
                        alice_basis,
                        eve_basis,
                    )
                    eve_sampler_requests.append(_sample_request(eve_circuit))
                    eve_locations.append((request_index, signal_index))
            alice_circuits.append(request_circuits)

        if not validated:
            return []

        eve_bits: dict[tuple[int, int], int] = {}
        if eve_sampler_requests:
            eve_results = require_sampler_results(
                self._sampler.run(eve_sampler_requests), eve_sampler_requests
            )
            for location, result in zip(eve_locations, eve_results):
                eve_bits[location] = _observed_bit(result.counts)

        bob_sampler_requests: list[SamplerRequest] = []
        result_slices: list[tuple[int, int]] = []
        for request_index, request in enumerate(validated):
            start = len(bob_sampler_requests)
            for signal_index in range(len(request.alice_bits)):
                eve_basis = (
                    None
                    if request.intercept_resend_bases is None
                    else request.intercept_resend_bases[signal_index]
                )
                if eve_basis is None:
                    circuit = alice_circuits[request_index][signal_index]
                else:
                    circuit = _build_prepare_measure_circuit(
                        eve_bits[(request_index, signal_index)],
                        eve_basis,
                        request.bob_bases[signal_index],
                    )
                bob_sampler_requests.append(_sample_request(circuit))
            result_slices.append((start, len(bob_sampler_requests)))

        bob_results = require_sampler_results(
            self._sampler.run(bob_sampler_requests), bob_sampler_requests
        )

        results: list[BB84Result] = []
        for request, (start, end) in zip(validated, result_slices):
            bob_bits = tuple(
                _observed_bit(result.counts) for result in bob_results[start:end]
            )
            sifted_indices = tuple(
                index
                for index, (alice_basis, bob_basis) in enumerate(
                    zip(request.alice_bases, request.bob_bases)
                )
                if alice_basis is bob_basis
            )
            results.append(
                BB84Result(
                    sifted_indices=sifted_indices,
                    parameter_estimation_indices=request.parameter_estimation_indices,
                    alice_sifted_bits=tuple(
                        request.alice_bits[index] for index in sifted_indices
                    ),
                    bob_sifted_bits=tuple(bob_bits[index] for index in sifted_indices),
                    max_qber=request.max_qber,
                )
            )
        return results
