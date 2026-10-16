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

"""Uniform public-contract coverage for every mixer component."""

import math

import pytest
from cqlib.circuit import ParameterError

from cqlib_algorithms.operators import BaseAnsatz
from cqlib_algorithms.operators.mixer import (
    ColorParityPermutationSwapMixer,
    ControlledBitFlipMixer,
    ControlledNullSwapMixer,
    ControlledSwapMixer,
    FullyConnectedMixer,
    MixerApplication,
    NearbyValuesMixer,
    NullSwapMixer,
    ParityRingMixer,
    PartitionedMixer,
    RingMixer,
    TimeColorMixer,
    TimeSwapMixer,
    ValueIndependentPermutationSwapMixer,
    ValueSelectivePermutationSwapMixer,
)
from cqlib_algorithms.protocols import Ansatz

MIXERS = (
    NearbyValuesMixer(3, 1),
    RingMixer(3),
    FullyConnectedMixer(3),
    NullSwapMixer(3),
    ParityRingMixer(3),
    ControlledBitFlipMixer(1),
    ControlledNullSwapMixer(1),
    ControlledSwapMixer(1),
    ValueSelectivePermutationSwapMixer(),
    ValueIndependentPermutationSwapMixer(2),
    ColorParityPermutationSwapMixer(2),
    TimeSwapMixer((1, 1), 0, 1, 0),
    TimeColorMixer((1, 1)),
    PartitionedMixer(
        2,
        [MixerApplication(ControlledBitFlipMixer(1), (0, 1))],
    ),
)


@pytest.mark.parametrize("mixer", MIXERS, ids=lambda mixer: type(mixer).__name__)
def test_every_public_mixer_has_the_uniform_ansatz_contract(mixer: Ansatz) -> None:
    assert isinstance(mixer, Ansatz)
    assert isinstance(mixer, BaseAnsatz)
    assert mixer.num_parameters() == 1

    first = mixer.build_circuit("contract")
    second = mixer.build_circuit("contract")
    assert mixer.num_qubits() == first.num_qubits
    assert first.used_symbols == ["contract_beta"]
    assert first == second
    assert first is not second


@pytest.mark.parametrize("mixer", MIXERS, ids=lambda mixer: type(mixer).__name__)
def test_every_public_mixer_accepts_empty_prefix_and_rejects_non_string(
    mixer: Ansatz,
) -> None:
    assert mixer.build_circuit("").used_symbols == ["_beta"]
    with pytest.raises(TypeError):
        mixer.build_circuit(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("mixer", MIXERS, ids=lambda mixer: type(mixer).__name__)
def test_every_public_mixer_rejects_nonfinite_beta_binding(mixer: Ansatz) -> None:
    circuit = mixer.build_circuit("contract")
    with pytest.raises(ParameterError):
        circuit.assign_parameters({"contract_beta": math.inf})
