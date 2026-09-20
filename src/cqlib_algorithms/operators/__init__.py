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

"""Operator subpackage for algorithmic quantum building blocks."""

from . import arithmetic
from .ansatz import (
    EfficientSU2Ansatz,
    HardwareEfficientAnsatz,
    PauliTwoDesignAnsatz,
    QAOACostEvolution,
    QAOAMixerEvolution,
    RealAmplitudesAnsatz,
    TwoLocalAnsatz,
)
from .oracle import (
    BitFlipOracle,
    BooleanFunction,
    BooleanVectorFunction,
    GeneralStateReflection,
    GroverDiffusion,
    PhaseOracle,
    ZeroStateReflection,
)
from .sampler import BackendSampler, StatevectorSampler
from .state_preparation import (
    AmplitudeEncoding,
    AngleEncoding,
    ArbitraryStatePreparation,
    BasisEncoding,
    BasisStateInitialization,
    FeatureMapPreparation,
    GHZStatePreparation,
    ProductStatePreparation,
    SparseAmplitudeEncoding,
    UniformSuperposition,
    UserCircuitPreparation,
)

__all__ = [
    "AmplitudeEncoding",
    "AngleEncoding",
    "ArbitraryStatePreparation",
    "BasisEncoding",
    "BasisStateInitialization",
    "BackendSampler",
    "BitFlipOracle",
    "EfficientSU2Ansatz",
    "BooleanFunction",
    "BooleanVectorFunction",
    "FeatureMapPreparation",
    "GHZStatePreparation",
    "GeneralStateReflection",
    "HardwareEfficientAnsatz",
    "GroverDiffusion",
    "PhaseOracle",
    "ProductStatePreparation",
    "PauliTwoDesignAnsatz",
    "QAOACostEvolution",
    "QAOAMixerEvolution",
    "RealAmplitudesAnsatz",
    "SparseAmplitudeEncoding",
    "StatevectorSampler",
    "TwoLocalAnsatz",
    "UniformSuperposition",
    "UserCircuitPreparation",
    "ZeroStateReflection",
    "arithmetic",
]
