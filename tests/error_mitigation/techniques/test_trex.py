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

"""Tests for twirled readout error extinction."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.circuit import StandardGate
from cqlib.device import ExecutionResult
from cqlib.qis import PauliString

from cqlib_algorithms.error_mitigation.core import QemExecutor
from cqlib_algorithms.error_mitigation.techniques.trex import (
    _build_circuits,
    _draw_twirls,
    combine_results,
    construct_circuits,
    execute_with_trex,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols import Sampler, SamplerRequest

_ATOL = 1e-9


def _twirl_offsets(num_randomizations: int, /) -> list[tuple[int, int]]:
    """Return the twirled and calibration offsets sharing one Pauli draw."""
    return [
        (index, index + num_randomizations)
        for index in range(num_randomizations)
    ]


def _twirl_mask(circuit: Circuit, body_length: int = 0, /) -> tuple[int, ...]:
    """Return one bit per qubit marking where an X gate sits.

    Parameters
    ----------
    circuit
        Circuit whose trailing operations carry the twirl Paulis.
    body_length
        Number of leading operations belonging to the underlying body, which
        are skipped so that only the appended twirl Paulis are read.
    """
    order = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    bits = [0] * circuit.num_qubits
    for operation in circuit.operations[body_length:]:
        bits[order[operation.qubits[0]]] = 1
    return tuple(bits)


class _ConfusedSampler(Sampler):
    """Sampler adding per-qubit readout flips to a noiseless run.

    The ideal distribution comes from a noiseless statevector sampler and is
    then convolved with the Kronecker product of the single-qubit confusion
    matrices, so the reported counts carry exactly the readout error the
    technique is meant to remove.
    """

    __slots__ = ("_inner", "_p0", "_p1")

    def __init__(self, p0: float, p1: float) -> None:
        self._inner = StatevectorSampler()
        self._p0 = p0
        self._p1 = p1

    def run(
        self,
        requests: Sequence[SamplerRequest],
        /,
    ) -> list[ExecutionResult]:
        """Return one confused count record per request, in input order."""
        results: list[ExecutionResult] = []
        for request, exact in zip(
            requests, self._inner.run(requests), strict=True
        ):
            width = exact.num_qubits
            ideal = np.zeros(1 << width, dtype=float)
            for key, count in exact.counts.items():
                ideal[int(key, 2)] += count
            ideal /= ideal.sum()

            # Bit i of the confusion index is qubit i, the convention the
            # little-endian count keys use.
            local = np.array(
                [[1.0 - self._p0, self._p1], [self._p0, 1.0 - self._p1]],
                dtype=float,
            )
            observed = reduce(np.kron, [local] * width) @ ideal
            observed /= observed.sum()

            draws = np.random.default_rng(request.seed).multinomial(
                request.shots, observed
            )
            counts = {
                format(index, f"0{width}b"): int(draws[index])
                for index in range(1 << width)
                if draws[index]
            }
            results.append(
                ExecutionResult.from_counts(
                    "confused-readout",
                    list(request.qubits),
                    request.shots,
                    width,
                    counts,
                )
            )
        return results


def _confusion_executor(p0: float, p1: float, /) -> QemExecutor:
    """Return an executor whose sampler carries a known readout confusion."""
    return QemExecutor(sampler=_ConfusedSampler(p0, p1))


def _bell_circuit() -> Circuit:
    """Return the two-qubit H-plus-CNOT body used for the readout tests."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def _zz_parity(counts: dict[str, int], shots: int, /) -> float:
    """Return the Z-basis parity of a count record as an expectation value."""
    return (
        sum(
            (-1) ** bitstring.count("1") * count
            for bitstring, count in counts.items()
        )
        / shots
    )


def test_construct_circuits_returns_two_times_num_randomizations_flat() -> None:
    circuit = _bell_circuit()
    num_randomizations = 6

    circuits = construct_circuits(
        circuit, num_randomizations=num_randomizations, seed=4
    )

    offsets = _twirl_offsets(num_randomizations)
    assert len(circuits) == 2 * num_randomizations
    for twirled, calibration in offsets:
        # The randomizations are laid out flat: the twirled copies first, then
        # their calibration partners, each calibration circuit holding nothing
        # but the Paulis of its own draw.
        assert len(circuits[calibration].operations) <= circuit.num_qubits
        assert len(circuits[twirled].operations) == len(
            circuit.operations
        ) + len(circuits[calibration].operations)

    with pytest.raises(ValueError, match="at least one"):
        construct_circuits(circuit, num_randomizations=0)


def test_the_two_halves_share_the_pauli_draw_of_their_offset() -> None:
    circuit = _bell_circuit()
    num_randomizations = 8

    circuits = construct_circuits(
        circuit, num_randomizations=num_randomizations, seed=13
    )

    for twirled, calibration in _twirl_offsets(num_randomizations):
        assert _twirl_mask(
            circuits[twirled], len(circuit.operations)
        ) == _twirl_mask(circuits[calibration])


def test_the_reconstruction_recovers_the_ideal_value_under_a_known_confusion() -> (
    None
):
    shots, num_randomizations = 20000, 16
    p0 = p1 = 0.05
    circuit = _bell_circuit()
    observable = PauliString.from_str("ZZ")
    executor = _confusion_executor(p0, p1)

    mitigated = execute_with_trex(
        circuit,
        executor,
        observable,
        shots=shots,
        num_randomizations=num_randomizations,
        seed=7,
    )
    sampled = executor.sample(
        [
            SamplerRequest(
                circuit=circuit,
                qubits=tuple(circuit.qubits),
                shots=shots,
                seed=7,
            )
        ]
    )[0]
    raw = _zz_parity(sampled.counts, shots)

    # Symmetric bit-flip readout scales the ideal value <ZZ> = 1 by
    # (1 - 2 p)^2, and dividing by the twirled calibration eigenvalue must
    # remove exactly that factor.
    assert mitigated == pytest.approx(1.0, abs=0.03)
    assert raw == pytest.approx((1.0 - 2.0 * p0) ** 2, abs=0.02)
    assert abs(mitigated - 1.0) < abs(raw - 1.0)


def test_identity_readout_is_a_no_op() -> None:
    shots = 4096
    twirls = [[0, 0], [1, 0], [0, 1], [1, 1]]
    observable = PauliString.from_str("ZZ")

    twirled_results: list[ExecutionResult] = []
    calibration_results: list[ExecutionResult] = []
    for index, row in enumerate(twirls):
        mask = sum(bit << qubit for qubit, bit in enumerate(row))
        # The twirl relabels the two Bell outcomes and the calibration
        # circuit reports the twirl itself, so the classical un-twirl maps
        # both back bit for bit.
        twirled = {
            format(0 ^ mask, "02b"): shots // 2,
            format(3 ^ mask, "02b"): shots - shots // 2,
        }
        calibration = {format(mask, "02b"): shots}
        twirled_results.append(
            ExecutionResult.from_counts(
                f"twirled-{index}", [0, 1], shots, 2, twirled
            )
        )
        calibration_results.append(
            ExecutionResult.from_counts(
                f"calibration-{index}", [0, 1], shots, 2, calibration
            )
        )

    mitigated = combine_results(
        [*twirled_results, *calibration_results],
        twirls,
        observable=observable,
    )

    # The calibration eigenvalue is one without readout error, so the
    # recombination returns the ideal value unchanged.
    assert mitigated == pytest.approx(1.0, abs=_ATOL)


def test_no_two_to_the_n_matrix_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
    exact_executor: QemExecutor,
) -> None:
    shapes: list[tuple[int, ...]] = []
    originals = {name: getattr(np.linalg, name) for name in ("inv", "pinv")}

    def spy(name: str):
        original = originals[name]

        def wrapper(matrix, *args, **kwargs):
            shapes.append(np.asarray(matrix).shape)
            return original(matrix, *args, **kwargs)

        return wrapper

    for name in originals:
        monkeypatch.setattr(np.linalg, name, spy(name))

    circuit = Circuit(5)
    for qubit in range(4):
        circuit.x(qubit)
    observable = PauliString.from_str("ZZZZZ")

    mitigated = execute_with_trex(
        circuit,
        exact_executor,
        observable,
        shots=64,
        num_randomizations=4,
        seed=9,
    )

    # |11110> has <Z^5> = 1, and ideal readout leaves the estimate exact.
    assert mitigated == pytest.approx(1.0, abs=_ATOL)
    # A readout matrix on five qubits would be a 32 x 32 inversion; the
    # correction must only ever touch per-qubit blocks.
    assert all(len(shape) < 2 or min(shape) <= 2 for shape in shapes), shapes


def test_the_twirl_is_symmetric_across_the_x_and_y_axes(
    exact_executor: QemExecutor,
) -> None:
    circuit = Circuit(2)
    circuit.x(0)
    observable = PauliString.from_str("ZZ")
    twirls = _draw_twirls(8, circuit.num_qubits, 11)

    estimates: list[float] = []
    for twirl_pauli in (StandardGate.X, StandardGate.Y):
        circuits = _build_circuits(circuit, twirls, twirl_pauli)
        results = exact_executor.sample(
            [
                SamplerRequest(
                    circuit=twirled_circuit,
                    qubits=tuple(twirled_circuit.qubits),
                    shots=1024,
                )
                for twirled_circuit in circuits
            ]
        )
        estimates.append(combine_results(results, twirls, observable=observable))

    # Both axes flip the recorded computational-basis bit, so one twirl table
    # drives the same recombination either way; |10> has <ZZ> = -1.
    assert estimates[0] == pytest.approx(-1.0, abs=_ATOL)
    assert estimates[1] == pytest.approx(estimates[0], abs=_ATOL)
