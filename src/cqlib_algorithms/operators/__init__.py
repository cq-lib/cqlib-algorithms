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
    BaseAnsatz,
    EfficientSU2Ansatz,
    HardwareEfficientAnsatz,
    PauliTwoDesignAnsatz,
    QAOACostEvolution,
    QAOAMixerEvolution,
    RealAmplitudesAnsatz,
    TwoLocalAnsatz,
)
from .arithmetic import QuadraticForm
from .composite import GroverOperator, HadamardPhaseQuery
from .estimator import SamplingEstimator, StatevectorEstimator
from .gradient import (
    FiniteDifferenceGradient,
    LinCombEstimatorGradient,
    NaturalGradient,
    ParameterShiftGradient,
    QFIResult,
    QGTResult,
    SPSAGradient,
    StatevectorQFI,
    StatevectorQGT,
)
from .measurement import (
    CommutingObservableGrouping,
    MeasurementGroup,
    MeasurementPlan,
    PauliGroupingStrategy,
    PauliMeasurementPlanner,
)
from .optimizer import Adam, GradientDescent, SPSAOptimizer
from .oracle import (
    BitFlipOracle,
    BooleanFunction,
    BooleanVectorFunction,
    GeneralStateReflection,
    GroverDiffusion,
    GroverLong,
    PhaseOracle,
    ZeroStateReflection,
)
from .sampler import BackendSampler, StatevectorSampler
from .state_fidelity import (
    CircuitFidelityRequest,
    CircuitFidelityResult,
    ComputeUncomputeFidelity,
)
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
    "Adam",
    "AmplitudeEncoding",
    "AngleEncoding",
    "ArbitraryStatePreparation",
    "BackendSampler",
    "BaseAnsatz",
    "BasisEncoding",
    "BasisStateInitialization",
    "BitFlipOracle",
    "BooleanFunction",
    "BooleanVectorFunction",
    "CircuitFidelityRequest",
    "CircuitFidelityResult",
    "CommutingObservableGrouping",
    "ComputeUncomputeFidelity",
    "EfficientSU2Ansatz",
    "FeatureMapPreparation",
    "FiniteDifferenceGradient",
    "GHZStatePreparation",
    "GeneralStateReflection",
    "GradientDescent",
    "GroverDiffusion",
    "GroverLong",
    "GroverOperator",
    "HadamardPhaseQuery",
    "HardwareEfficientAnsatz",
    "LinCombEstimatorGradient",
    "MeasurementGroup",
    "MeasurementPlan",
    "NaturalGradient",
    "ParameterShiftGradient",
    "PauliGroupingStrategy",
    "PauliMeasurementPlanner",
    "PauliTwoDesignAnsatz",
    "PhaseOracle",
    "ProductStatePreparation",
    "QAOACostEvolution",
    "QAOAMixerEvolution",
    "QFIResult",
    "QGTResult",
    "QuadraticForm",
    "RealAmplitudesAnsatz",
    "SPSAGradient",
    "SPSAOptimizer",
    "SamplingEstimator",
    "SparseAmplitudeEncoding",
    "StatevectorEstimator",
    "StatevectorQFI",
    "StatevectorQGT",
    "StatevectorSampler",
    "TwoLocalAnsatz",
    "UniformSuperposition",
    "UserCircuitPreparation",
    "ZeroStateReflection",
    "arithmetic",
]
