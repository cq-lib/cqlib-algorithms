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

"""Smoke checks for the installed distribution."""

import inspect
from importlib import import_module
from importlib.metadata import metadata
from importlib.resources import files

import pytest
from cqlib.error_mitigation import Estimator as ErrorMitigationEstimator

import cqlib_algorithms
from cqlib_algorithms import (
    BB84,
    CHSH,
    QAOA,
    QFT,
    VQE,
    AdaptVQE,
    AdaptVQEResult,
    AdaptVQETermination,
    AmplitudeEstimationProblem,
    BB84Basis,
    BB84Request,
    BB84Result,
    BernsteinVazirani,
    BernsteinVaziraniRequest,
    BernsteinVaziraniResult,
    CHSHRequest,
    CHSHResult,
    DeutschJozsa,
    DeutschJozsaRequest,
    DeutschJozsaResult,
    ElectronicGroundState,
    ElectronicGroundStateResult,
    ElectronicQEOM,
    ElectronicQEOMResult,
    FasterAmplitudeEstimation,
    FasterAmplitudeEstimationRequest,
    FasterAmplitudeEstimationResult,
    HamiltonianPhaseEstimation,
    HamiltonianPhaseEstimationRequest,
    HamiltonianPhaseEstimationResult,
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationRequest,
    IterativeAmplitudeEstimationResult,
    IterativePhaseEstimation,
    IterativePhaseEstimationRequest,
    IterativePhaseEstimationResult,
    LatticeGroundState,
    LatticeGroundStateResult,
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationRequest,
    MaximumLikelihoodAmplitudeEstimationResult,
    PhaseEstimation,
    PhaseEstimationCircuit,
    PhaseEstimationProblem,
    PhaseEstimationRequest,
    PhaseEstimationResult,
    QAOAResult,
    QuantumTeleportation,
    QuantumTeleportationRequest,
    QuantumTeleportationResult,
    SamplingVQE,
    SamplingVQEMeasurement,
    SamplingVQEResult,
    Simon,
    SimonRequest,
    SimonResult,
    StandardAmplitudeEstimation,
    StandardAmplitudeEstimationRequest,
    StandardAmplitudeEstimationResult,
    SuperdenseCoding,
    SuperdenseCodingRequest,
    SuperdenseCodingResult,
    VibrationalGroundState,
    VibrationalGroundStateResult,
    VibrationalMeasurements,
    VQAResult,
    VQEResult,
)
from cqlib_algorithms.algorithms import BB84 as AlgorithmBB84
from cqlib_algorithms.algorithms import CHSH as AlgorithmCHSH
from cqlib_algorithms.algorithms import QAOA as AlgorithmQAOA
from cqlib_algorithms.algorithms import VQE as AlgorithmVQE
from cqlib_algorithms.algorithms import AdaptVQE as AlgorithmAdaptVQE
from cqlib_algorithms.algorithms import (
    AdaptVQEResult as AlgorithmAdaptVQEResult,
)
from cqlib_algorithms.algorithms import (
    AdaptVQETermination as AlgorithmAdaptVQETermination,
)
from cqlib_algorithms.algorithms import (
    AmplitudeEstimationProblem as AlgorithmAmplitudeEstimationProblem,
)
from cqlib_algorithms.algorithms import BB84Basis as AlgorithmBB84Basis
from cqlib_algorithms.algorithms import BB84Request as AlgorithmBB84Request
from cqlib_algorithms.algorithms import BB84Result as AlgorithmBB84Result
from cqlib_algorithms.algorithms import (
    BernsteinVazirani as AlgorithmBernsteinVazirani,
)
from cqlib_algorithms.algorithms import (
    BernsteinVaziraniRequest as AlgorithmBernsteinVaziraniRequest,
)
from cqlib_algorithms.algorithms import (
    BernsteinVaziraniResult as AlgorithmBernsteinVaziraniResult,
)
from cqlib_algorithms.algorithms import CHSHRequest as AlgorithmCHSHRequest
from cqlib_algorithms.algorithms import CHSHResult as AlgorithmCHSHResult
from cqlib_algorithms.algorithms import (
    DeutschJozsa as AlgorithmDeutschJozsa,
)
from cqlib_algorithms.algorithms import (
    DeutschJozsaRequest as AlgorithmDeutschJozsaRequest,
)
from cqlib_algorithms.algorithms import (
    DeutschJozsaResult as AlgorithmDeutschJozsaResult,
)
from cqlib_algorithms.algorithms import (
    ElectronicGroundState as AlgorithmElectronicGroundState,
)
from cqlib_algorithms.algorithms import (
    ElectronicGroundStateResult as AlgorithmElectronicGroundStateResult,
)
from cqlib_algorithms.algorithms import ElectronicQEOM as AlgorithmElectronicQEOM
from cqlib_algorithms.algorithms import (
    ElectronicQEOMResult as AlgorithmElectronicQEOMResult,
)
from cqlib_algorithms.algorithms import (
    FasterAmplitudeEstimation as AlgorithmFasterAmplitudeEstimation,
)
from cqlib_algorithms.algorithms import (
    FasterAmplitudeEstimationRequest as AlgorithmFasterAmplitudeEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    FasterAmplitudeEstimationResult as AlgorithmFasterAmplitudeEstimationResult,
)
from cqlib_algorithms.algorithms import (
    HamiltonianPhaseEstimation as AlgorithmHamiltonianPhaseEstimation,
)
from cqlib_algorithms.algorithms import (
    HamiltonianPhaseEstimationRequest as AlgorithmHamiltonianPhaseEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    HamiltonianPhaseEstimationResult as AlgorithmHamiltonianPhaseEstimationResult,
)
from cqlib_algorithms.algorithms import (
    IterativeAmplitudeEstimation as AlgorithmIterativeAmplitudeEstimation,
)
from cqlib_algorithms.algorithms import (
    IterativeAmplitudeEstimationRequest as AlgorithmIterativeAmplitudeEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    IterativeAmplitudeEstimationResult as AlgorithmIterativeAmplitudeEstimationResult,
)
from cqlib_algorithms.algorithms import (
    IterativePhaseEstimation as AlgorithmIterativePhaseEstimation,
)
from cqlib_algorithms.algorithms import (
    IterativePhaseEstimationRequest as AlgorithmIterativePhaseEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    IterativePhaseEstimationResult as AlgorithmIterativePhaseEstimationResult,
)
from cqlib_algorithms.algorithms import (
    LatticeGroundState as AlgorithmLatticeGroundState,
)
from cqlib_algorithms.algorithms import (
    LatticeGroundStateResult as AlgorithmLatticeGroundStateResult,
)
from cqlib_algorithms.algorithms import (
    MaximumLikelihoodAmplitudeEstimation as AlgorithmMaximumLikelihoodAmplitudeEstimation,
)
from cqlib_algorithms.algorithms import (
    MaximumLikelihoodAmplitudeEstimationRequest as AlgorithmMaximumLikelihoodAmplitudeEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    MaximumLikelihoodAmplitudeEstimationResult as AlgorithmMaximumLikelihoodAmplitudeEstimationResult,
)
from cqlib_algorithms.algorithms import PhaseEstimation as AlgorithmPhaseEstimation
from cqlib_algorithms.algorithms import (
    PhaseEstimationProblem as AlgorithmPhaseEstimationProblem,
)
from cqlib_algorithms.algorithms import (
    PhaseEstimationRequest as AlgorithmPhaseEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    PhaseEstimationResult as AlgorithmPhaseEstimationResult,
)
from cqlib_algorithms.algorithms import QAOAResult as AlgorithmQAOAResult
from cqlib_algorithms.algorithms import (
    QuantumTeleportation as AlgorithmQuantumTeleportation,
)
from cqlib_algorithms.algorithms import (
    QuantumTeleportationRequest as AlgorithmQuantumTeleportationRequest,
)
from cqlib_algorithms.algorithms import (
    QuantumTeleportationResult as AlgorithmQuantumTeleportationResult,
)
from cqlib_algorithms.algorithms import SamplingVQE as AlgorithmSamplingVQE
from cqlib_algorithms.algorithms import (
    SamplingVQEMeasurement as AlgorithmSamplingVQEMeasurement,
)
from cqlib_algorithms.algorithms import (
    SamplingVQEResult as AlgorithmSamplingVQEResult,
)
from cqlib_algorithms.algorithms import Simon as AlgorithmSimon
from cqlib_algorithms.algorithms import SimonRequest as AlgorithmSimonRequest
from cqlib_algorithms.algorithms import SimonResult as AlgorithmSimonResult
from cqlib_algorithms.algorithms import (
    StandardAmplitudeEstimation as AlgorithmStandardAmplitudeEstimation,
)
from cqlib_algorithms.algorithms import (
    StandardAmplitudeEstimationRequest as AlgorithmStandardAmplitudeEstimationRequest,
)
from cqlib_algorithms.algorithms import (
    StandardAmplitudeEstimationResult as AlgorithmStandardAmplitudeEstimationResult,
)
from cqlib_algorithms.algorithms import (
    SuperdenseCoding as AlgorithmSuperdenseCoding,
)
from cqlib_algorithms.algorithms import (
    SuperdenseCodingRequest as AlgorithmSuperdenseCodingRequest,
)
from cqlib_algorithms.algorithms import (
    SuperdenseCodingResult as AlgorithmSuperdenseCodingResult,
)
from cqlib_algorithms.algorithms import (
    VibrationalGroundState as AlgorithmVibrationalGroundState,
)
from cqlib_algorithms.algorithms import (
    VibrationalGroundStateResult as AlgorithmVibrationalGroundStateResult,
)
from cqlib_algorithms.algorithms import (
    VibrationalMeasurements as AlgorithmVibrationalMeasurements,
)
from cqlib_algorithms.algorithms import VQAResult as AlgorithmVQAResult
from cqlib_algorithms.algorithms import VQEResult as AlgorithmVQEResult
from cqlib_algorithms.algorithms.variational import QAOA as VariationalQAOA
from cqlib_algorithms.algorithms.variational import VQE as VariationalVQE
from cqlib_algorithms.algorithms.variational import (
    AdaptVQE as VariationalAdaptVQE,
)
from cqlib_algorithms.algorithms.variational import (
    AdaptVQEResult as VariationalAdaptVQEResult,
)
from cqlib_algorithms.algorithms.variational import (
    AdaptVQETermination as VariationalAdaptVQETermination,
)
from cqlib_algorithms.algorithms.variational import (
    QAOAResult as VariationalQAOAResult,
)
from cqlib_algorithms.algorithms.variational import (
    SamplingVQE as VariationalSamplingVQE,
)
from cqlib_algorithms.algorithms.variational import (
    SamplingVQEMeasurement as VariationalSamplingVQEMeasurement,
)
from cqlib_algorithms.algorithms.variational import (
    SamplingVQEResult as VariationalSamplingVQEResult,
)
from cqlib_algorithms.algorithms.variational import (
    VQAResult as VariationalVQAResult,
)
from cqlib_algorithms.algorithms.variational import (
    VQEResult as VariationalVQEResult,
)
from cqlib_algorithms.operators import (
    PhaseEstimationCircuit as OperatorPhaseEstimationCircuit,
)
from cqlib_algorithms.protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Gradient,
    GradientRequest,
    GradientResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
    Sampler,
    SamplerRequest,
    StatePreparation,
)
from cqlib_algorithms.transforms import QFT as TransformQFT

PUBLIC_COMPONENTS = {
    "AdaptVQE": AdaptVQE,
    "AdaptVQEResult": AdaptVQEResult,
    "AdaptVQETermination": AdaptVQETermination,
    "AmplitudeEstimationProblem": AmplitudeEstimationProblem,
    "Ansatz": Ansatz,
    "BB84": BB84,
    "BB84Basis": BB84Basis,
    "BB84Request": BB84Request,
    "BB84Result": BB84Result,
    "BernsteinVazirani": BernsteinVazirani,
    "BernsteinVaziraniRequest": BernsteinVaziraniRequest,
    "BernsteinVaziraniResult": BernsteinVaziraniResult,
    "CHSH": CHSH,
    "CHSHRequest": CHSHRequest,
    "CHSHResult": CHSHResult,
    "DeutschJozsa": DeutschJozsa,
    "DeutschJozsaRequest": DeutschJozsaRequest,
    "DeutschJozsaResult": DeutschJozsaResult,
    "ElectronicGroundState": ElectronicGroundState,
    "ElectronicGroundStateResult": ElectronicGroundStateResult,
    "ElectronicQEOM": ElectronicQEOM,
    "ElectronicQEOMResult": ElectronicQEOMResult,
    "FasterAmplitudeEstimation": FasterAmplitudeEstimation,
    "FasterAmplitudeEstimationRequest": FasterAmplitudeEstimationRequest,
    "FasterAmplitudeEstimationResult": FasterAmplitudeEstimationResult,
    "HamiltonianPhaseEstimation": HamiltonianPhaseEstimation,
    "HamiltonianPhaseEstimationRequest": HamiltonianPhaseEstimationRequest,
    "HamiltonianPhaseEstimationResult": HamiltonianPhaseEstimationResult,
    "IterativeAmplitudeEstimation": IterativeAmplitudeEstimation,
    "IterativeAmplitudeEstimationRequest": IterativeAmplitudeEstimationRequest,
    "IterativeAmplitudeEstimationResult": IterativeAmplitudeEstimationResult,
    "IterativePhaseEstimation": IterativePhaseEstimation,
    "IterativePhaseEstimationRequest": IterativePhaseEstimationRequest,
    "IterativePhaseEstimationResult": IterativePhaseEstimationResult,
    "LatticeGroundState": LatticeGroundState,
    "LatticeGroundStateResult": LatticeGroundStateResult,
    "MaximumLikelihoodAmplitudeEstimation": MaximumLikelihoodAmplitudeEstimation,
    "MaximumLikelihoodAmplitudeEstimationRequest": MaximumLikelihoodAmplitudeEstimationRequest,
    "MaximumLikelihoodAmplitudeEstimationResult": MaximumLikelihoodAmplitudeEstimationResult,
    "Estimator": Estimator,
    "EstimatorRequest": EstimatorRequest,
    "EstimatorResult": EstimatorResult,
    "Gradient": Gradient,
    "GradientRequest": GradientRequest,
    "GradientResult": GradientResult,
    "Optimizer": Optimizer,
    "OptimizerRequest": OptimizerRequest,
    "OptimizerResult": OptimizerResult,
    "OptimizerStatus": OptimizerStatus,
    "QAOA": QAOA,
    "QAOAResult": QAOAResult,
    "PhaseEstimation": PhaseEstimation,
    "PhaseEstimationCircuit": PhaseEstimationCircuit,
    "PhaseEstimationProblem": PhaseEstimationProblem,
    "PhaseEstimationRequest": PhaseEstimationRequest,
    "PhaseEstimationResult": PhaseEstimationResult,
    "QuantumTeleportation": QuantumTeleportation,
    "QuantumTeleportationRequest": QuantumTeleportationRequest,
    "QuantumTeleportationResult": QuantumTeleportationResult,
    "Sampler": Sampler,
    "SamplerRequest": SamplerRequest,
    "SamplingVQE": SamplingVQE,
    "SamplingVQEMeasurement": SamplingVQEMeasurement,
    "SamplingVQEResult": SamplingVQEResult,
    "Simon": Simon,
    "SimonRequest": SimonRequest,
    "SimonResult": SimonResult,
    "StatePreparation": StatePreparation,
    "SuperdenseCoding": SuperdenseCoding,
    "SuperdenseCodingRequest": SuperdenseCodingRequest,
    "SuperdenseCodingResult": SuperdenseCodingResult,
    "StandardAmplitudeEstimation": StandardAmplitudeEstimation,
    "StandardAmplitudeEstimationRequest": StandardAmplitudeEstimationRequest,
    "StandardAmplitudeEstimationResult": StandardAmplitudeEstimationResult,
    "VQE": VQE,
    "VQEResult": VQEResult,
    "VQAResult": VQAResult,
    "VibrationalGroundState": VibrationalGroundState,
    "VibrationalGroundStateResult": VibrationalGroundStateResult,
    "VibrationalMeasurements": VibrationalMeasurements,
}


def test_distribution_metadata():
    distribution = metadata("cqlib-algorithms")
    assert distribution["Requires-Python"] == ">=3.10"
    assert cqlib_algorithms.__version__ == distribution["Version"]


def test_algorithms_package_is_importable():
    assert import_module("cqlib_algorithms.algorithms") is not None


def test_protocols_package_is_importable():
    assert import_module("cqlib_algorithms.protocols") is not None


def test_transforms_package_is_importable():
    assert import_module("cqlib_algorithms.transforms") is not None


def test_typing_marker_is_packaged():
    assert files("cqlib_algorithms").joinpath("py.typed").is_file()


@pytest.mark.parametrize(("name", "component"), PUBLIC_COMPONENTS.items())
def test_stable_components_are_exported_from_package(name, component):
    assert getattr(cqlib_algorithms, name) is component


def test_package_all_matches_the_stable_public_api():
    assert set(cqlib_algorithms.__all__) == {"__version__", "QFT", *PUBLIC_COMPONENTS}


def test_estimator_contract_is_distinct_from_the_core_callback_alias():
    assert cqlib_algorithms.Estimator is not ErrorMitigationEstimator


def test_component_contracts_are_abstract_base_classes():
    for component in (
        Ansatz,
        Estimator,
        Gradient,
        Optimizer,
        Sampler,
        StatePreparation,
    ):
        assert inspect.isabstract(component)


def test_qft_is_exported_from_public_namespaces():
    assert QFT is TransformQFT


def test_phase_estimation_is_exported_from_public_namespaces():
    assert PhaseEstimationCircuit is OperatorPhaseEstimationCircuit
    assert PhaseEstimation is AlgorithmPhaseEstimation
    assert PhaseEstimationProblem is AlgorithmPhaseEstimationProblem
    assert PhaseEstimationRequest is AlgorithmPhaseEstimationRequest
    assert PhaseEstimationResult is AlgorithmPhaseEstimationResult
    assert IterativePhaseEstimation is AlgorithmIterativePhaseEstimation
    assert IterativePhaseEstimationRequest is AlgorithmIterativePhaseEstimationRequest
    assert IterativePhaseEstimationResult is AlgorithmIterativePhaseEstimationResult
    assert HamiltonianPhaseEstimation is AlgorithmHamiltonianPhaseEstimation
    assert (
        HamiltonianPhaseEstimationRequest is AlgorithmHamiltonianPhaseEstimationRequest
    )
    assert HamiltonianPhaseEstimationResult is AlgorithmHamiltonianPhaseEstimationResult


def test_vqe_is_exported_from_public_namespaces():
    assert VQE is AlgorithmVQE is VariationalVQE
    assert VQEResult is AlgorithmVQEResult is VariationalVQEResult
    assert VQAResult is AlgorithmVQAResult is VariationalVQAResult


def test_adaptive_and_sampling_vqe_are_exported_from_public_namespaces():
    assert AdaptVQE is AlgorithmAdaptVQE is VariationalAdaptVQE
    assert AdaptVQEResult is AlgorithmAdaptVQEResult is VariationalAdaptVQEResult
    assert (
        AdaptVQETermination
        is AlgorithmAdaptVQETermination
        is VariationalAdaptVQETermination
    )
    assert SamplingVQE is AlgorithmSamplingVQE is VariationalSamplingVQE
    assert (
        SamplingVQEMeasurement
        is AlgorithmSamplingVQEMeasurement
        is VariationalSamplingVQEMeasurement
    )
    assert (
        SamplingVQEResult is AlgorithmSamplingVQEResult is VariationalSamplingVQEResult
    )


def test_electronic_ground_state_is_exported_from_public_namespaces():
    assert ElectronicGroundState is AlgorithmElectronicGroundState
    assert ElectronicGroundStateResult is AlgorithmElectronicGroundStateResult
    assert ElectronicQEOM is AlgorithmElectronicQEOM
    assert ElectronicQEOMResult is AlgorithmElectronicQEOMResult


def test_lattice_ground_state_is_exported_from_public_namespaces():
    assert LatticeGroundState is AlgorithmLatticeGroundState
    assert LatticeGroundStateResult is AlgorithmLatticeGroundStateResult


def test_vibrational_ground_state_is_exported_from_public_namespaces():
    assert VibrationalGroundState is AlgorithmVibrationalGroundState
    assert VibrationalGroundStateResult is AlgorithmVibrationalGroundStateResult
    assert VibrationalMeasurements is AlgorithmVibrationalMeasurements


def test_amplitude_estimation_is_exported_from_public_namespaces():
    assert AmplitudeEstimationProblem is AlgorithmAmplitudeEstimationProblem
    assert FasterAmplitudeEstimation is AlgorithmFasterAmplitudeEstimation
    assert FasterAmplitudeEstimationRequest is AlgorithmFasterAmplitudeEstimationRequest
    assert FasterAmplitudeEstimationResult is AlgorithmFasterAmplitudeEstimationResult
    assert StandardAmplitudeEstimation is AlgorithmStandardAmplitudeEstimation
    assert (
        StandardAmplitudeEstimationRequest
        is AlgorithmStandardAmplitudeEstimationRequest
    )
    assert (
        StandardAmplitudeEstimationResult is AlgorithmStandardAmplitudeEstimationResult
    )
    assert IterativeAmplitudeEstimation is AlgorithmIterativeAmplitudeEstimation
    assert (
        IterativeAmplitudeEstimationRequest
        is AlgorithmIterativeAmplitudeEstimationRequest
    )
    assert (
        IterativeAmplitudeEstimationResult
        is AlgorithmIterativeAmplitudeEstimationResult
    )
    assert (
        MaximumLikelihoodAmplitudeEstimation
        is AlgorithmMaximumLikelihoodAmplitudeEstimation
    )
    assert (
        MaximumLikelihoodAmplitudeEstimationRequest
        is AlgorithmMaximumLikelihoodAmplitudeEstimationRequest
    )
    assert (
        MaximumLikelihoodAmplitudeEstimationResult
        is AlgorithmMaximumLikelihoodAmplitudeEstimationResult
    )


def test_qaoa_is_exported_from_public_namespaces():
    assert QAOA is AlgorithmQAOA is VariationalQAOA
    assert QAOAResult is AlgorithmQAOAResult is VariationalQAOAResult


def test_quantum_teleportation_is_exported_from_public_namespaces():
    assert QuantumTeleportation is AlgorithmQuantumTeleportation
    assert QuantumTeleportationRequest is AlgorithmQuantumTeleportationRequest
    assert QuantumTeleportationResult is AlgorithmQuantumTeleportationResult


def test_deutsch_jozsa_is_exported_from_public_namespaces():
    assert DeutschJozsa is AlgorithmDeutschJozsa
    assert DeutschJozsaRequest is AlgorithmDeutschJozsaRequest
    assert DeutschJozsaResult is AlgorithmDeutschJozsaResult


def test_bernstein_vazirani_is_exported_from_public_namespaces():
    assert BernsteinVazirani is AlgorithmBernsteinVazirani
    assert BernsteinVaziraniRequest is AlgorithmBernsteinVaziraniRequest
    assert BernsteinVaziraniResult is AlgorithmBernsteinVaziraniResult


def test_chsh_is_exported_from_public_namespaces():
    assert CHSH is AlgorithmCHSH
    assert CHSHRequest is AlgorithmCHSHRequest
    assert CHSHResult is AlgorithmCHSHResult


def test_bb84_is_exported_from_public_namespaces():
    assert BB84 is AlgorithmBB84
    assert BB84Basis is AlgorithmBB84Basis
    assert BB84Request is AlgorithmBB84Request
    assert BB84Result is AlgorithmBB84Result


def test_simon_is_exported_from_public_namespaces():
    assert Simon is AlgorithmSimon
    assert SimonRequest is AlgorithmSimonRequest
    assert SimonResult is AlgorithmSimonResult


def test_superdense_coding_is_exported_from_public_namespaces():
    assert SuperdenseCoding is AlgorithmSuperdenseCoding
    assert SuperdenseCodingRequest is AlgorithmSuperdenseCodingRequest
    assert SuperdenseCodingResult is AlgorithmSuperdenseCodingResult
