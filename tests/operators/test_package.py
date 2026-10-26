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

"""Smoke checks for the public operators packages."""

from importlib import import_module

from cqlib_algorithms.protocols import (
    Ansatz,
    Estimator,
    Gradient,
    Optimizer,
    Sampler,
    StatePreparation,
)


def test_state_preparation_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    state_preparation = import_module("cqlib_algorithms.operators.state_preparation")
    expected = {
        "AmplitudeEncoding",
        "AngleEncoding",
        "ArbitraryStatePreparation",
        "BasisEncoding",
        "BasisStateInitialization",
        "BRQI",
        "DickeStatePreparation",
        "FeatureMapPreparation",
        "GHZStatePreparation",
        "ImageAngleEncoding",
        "ImageBasisEncoding",
        "ProductStatePreparation",
        "SparseAmplitudeEncoding",
        "UniformSuperposition",
        "UserCircuitPreparation",
    }
    assert expected <= set(operators.__all__)
    assert set(state_preparation.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(state_preparation, name)
        assert issubclass(getattr(state_preparation, name), StatePreparation)
    assert operators.BRQI is state_preparation.BRQI
    assert operators.BRQI is state_preparation.ImageBasisEncoding


def test_oracle_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    oracle = import_module("cqlib_algorithms.operators.oracle")
    state_preparation = import_module("cqlib_algorithms.operators.state_preparation")
    expected = {
        "BlockEncoding",
        "BitFlipOracle",
        "BooleanFunction",
        "BooleanVectorFunction",
        "DickeUnitary",
        "GeneralStateReflection",
        "GroverDiffusion",
        "GroverLong",
        "PhaseOracle",
        "SplitCycleShift",
        "UnitaryDilation",
        "UnitaryMatrixOracle",
        "WeightDistributionBlock",
        "ZeroStateReflection",
    }

    assert set(oracle.__all__) == expected
    assert expected | set(state_preparation.__all__) <= set(operators.__all__)
    for name in expected:
        assert getattr(operators, name) is getattr(oracle, name)


def test_composite_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    composite = import_module("cqlib_algorithms.operators.composite")
    expected = {
        "GroverOperator",
        "HadamardPhaseQuery",
        "HiddenLinearFunctionCircuit",
        "PhaseEstimationCircuit",
        "ProjectorControlledPhase",
        "QuantumSignalProcessing",
        "QuantumSingularValueTransformation",
        "RealValuedQSVT",
    }

    assert expected <= set(operators.__all__)
    assert set(composite.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(composite, name)


def test_ansatz_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    ansatz = import_module("cqlib_algorithms.operators.ansatz")
    concrete = {
        "EfficientSU2Ansatz",
        "EvolvedOperatorAnsatz",
        "ExcitationPreservingAnsatz",
        "HamiltonianVariationalAnsatz",
        "HardwareEfficientAnsatz",
        "MERAAnsatz",
        "MPSAnsatz",
        "PauliTwoDesignAnsatz",
        "QAOACostEvolution",
        "QAOAMixerEvolution",
        "RandomLayersAnsatz",
        "RealAmplitudesAnsatz",
        "SimplifiedTwoDesignAnsatz",
        "TwoLocalAnsatz",
        "TwoQubitUnitaryAnsatz",
        "TTNAnsatz",
    }
    expected = concrete | {"BaseAnsatz"}
    assert expected <= set(operators.__all__)
    assert set(ansatz.__all__) == expected
    assert operators.BaseAnsatz is ansatz.BaseAnsatz
    for name in concrete:
        assert getattr(operators, name) is getattr(ansatz, name)
        assert issubclass(getattr(ansatz, name), Ansatz)
        assert issubclass(getattr(ansatz, name), ansatz.BaseAnsatz)


def test_mixer_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    mixer = import_module("cqlib_algorithms.operators.mixer")
    concrete = {
        "ColorParityPermutationSwapMixer",
        "ControlledBitFlipMixer",
        "ControlledNullSwapMixer",
        "ControlledSwapMixer",
        "FullyConnectedMixer",
        "NearbyValuesMixer",
        "NullSwapMixer",
        "ParityRingMixer",
        "PartitionedMixer",
        "RingMixer",
        "TimeColorMixer",
        "TimeSwapMixer",
        "ValueIndependentPermutationSwapMixer",
        "ValueSelectivePermutationSwapMixer",
    }
    expected = concrete | {"MixerApplication"}
    assert expected <= set(operators.__all__)
    assert set(mixer.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(mixer, name)
    for name in concrete:
        assert issubclass(getattr(mixer, name), Ansatz)
        assert issubclass(getattr(mixer, name), operators.BaseAnsatz)


def test_measurement_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    measurement = import_module("cqlib_algorithms.operators.measurement")
    expected = {
        "CommutingObservableGrouping",
        "MeasurementGroup",
        "MeasurementPlan",
        "PauliGroupingStrategy",
        "PauliMeasurementPlanner",
    }

    assert expected <= set(operators.__all__)
    assert set(measurement.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(measurement, name)


def test_information_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    information = import_module("cqlib_algorithms.operators.information")
    expected = {"mutual_information"}

    assert expected <= set(operators.__all__)
    assert set(information.__all__) == expected
    assert operators.mutual_information is information.mutual_information


def test_sampler_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    sampler = import_module("cqlib_algorithms.operators.sampler")
    expected = {"BackendSampler", "StatevectorSampler"}

    assert expected <= set(operators.__all__)
    assert set(sampler.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(sampler, name)
        assert issubclass(getattr(sampler, name), Sampler)


def test_estimator_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    estimator = import_module("cqlib_algorithms.operators.estimator")
    expected = {"SamplingEstimator", "StatevectorEstimator"}

    assert expected <= set(operators.__all__)
    assert set(estimator.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(estimator, name)
        assert issubclass(getattr(estimator, name), Estimator)


def test_state_fidelity_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    state_fidelity = import_module("cqlib_algorithms.operators.state_fidelity")
    expected = {
        "CircuitFidelityRequest",
        "CircuitFidelityResult",
        "ComputeUncomputeFidelity",
        "UnitaryOverlap",
    }

    assert expected <= set(operators.__all__)
    assert set(state_fidelity.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(state_fidelity, name)


def test_gradient_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    gradient = import_module("cqlib_algorithms.operators.gradient")
    concrete_gradients = {
        "FiniteDifferenceGradient",
        "LinCombEstimatorGradient",
        "ParameterShiftGradient",
        "SPSAGradient",
    }
    expected = concrete_gradients | {
        "NaturalGradient",
        "QFIResult",
        "QGTResult",
        "StatevectorQFI",
        "StatevectorQGT",
    }

    assert expected <= set(operators.__all__)
    assert set(gradient.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(gradient, name)
    for name in concrete_gradients:
        assert issubclass(getattr(gradient, name), Gradient)


def test_optimizer_operators_are_exported_from_functional_packages():
    operators = import_module("cqlib_algorithms.operators")
    optimizer = import_module("cqlib_algorithms.operators.optimizer")
    expected = {
        "Adam",
        "COBYLAOptimizer",
        "GradientDescent",
        "LBFGSBOptimizer",
        "SPSAOptimizer",
    }

    assert expected <= set(operators.__all__)
    assert set(optimizer.__all__) == expected
    for name in expected:
        assert getattr(operators, name) is getattr(optimizer, name)
        assert issubclass(getattr(optimizer, name), Optimizer)
        assert Optimizer in getattr(optimizer, name).__bases__
