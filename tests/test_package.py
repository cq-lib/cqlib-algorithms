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
    PVQD,
    QAOA,
    QFT,
    QSVR,
    SQD,
    VQD,
    VQE,
    VQR,
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
    Grover,
    GroverRequest,
    GroverResult,
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
    PegasosQSVC,
    PhaseEstimation,
    PhaseEstimationCircuit,
    PhaseEstimationProblem,
    PhaseEstimationRequest,
    PhaseEstimationResult,
    PVQDResult,
    QAOAResult,
    QBayesian,
    QBayesianResult,
    QBayesianRound,
    QBayesianStatus,
    QuantumKernelTrainer,
    QuantumKernelTrainerResult,
    QuantumTeleportation,
    QuantumTeleportationRequest,
    QuantumTeleportationResult,
    SamplingVQE,
    SamplingVQEMeasurement,
    SamplingVQEResult,
    Simon,
    SimonRequest,
    SimonResult,
    SQDCallback,
    SQDIteration,
    SQDResult,
    SQDSubspaceResult,
    SQDSubspaceSolver,
    StandardAmplitudeEstimation,
    StandardAmplitudeEstimationRequest,
    StandardAmplitudeEstimationResult,
    SuperdenseCoding,
    SuperdenseCodingRequest,
    SuperdenseCodingResult,
    VarQITE,
    VarQITEResult,
    VarQRTE,
    VarQRTEResult,
    VibrationalGroundState,
    VibrationalGroundStateResult,
    VibrationalMeasurements,
    VQAResult,
    VQDResult,
    VQEResult,
    recover_sqd_configurations,
    solve_sqd_subspace,
)
from cqlib_algorithms.algorithms import BB84 as AlgorithmBB84
from cqlib_algorithms.algorithms import CHSH as AlgorithmCHSH
from cqlib_algorithms.algorithms import PVQD as AlgorithmPVQD
from cqlib_algorithms.algorithms import QAOA as AlgorithmQAOA
from cqlib_algorithms.algorithms import SQD as AlgorithmSQD
from cqlib_algorithms.algorithms import VQD as AlgorithmVQD
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
from cqlib_algorithms.algorithms import Grover as AlgorithmGrover
from cqlib_algorithms.algorithms import GroverRequest as AlgorithmGroverRequest
from cqlib_algorithms.algorithms import GroverResult as AlgorithmGroverResult
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
from cqlib_algorithms.algorithms import PVQDResult as AlgorithmPVQDResult
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
from cqlib_algorithms.algorithms import SQDCallback as AlgorithmSQDCallback
from cqlib_algorithms.algorithms import SQDIteration as AlgorithmSQDIteration
from cqlib_algorithms.algorithms import SQDResult as AlgorithmSQDResult
from cqlib_algorithms.algorithms import (
    SQDSubspaceResult as AlgorithmSQDSubspaceResult,
)
from cqlib_algorithms.algorithms import SQDSubspaceSolver as AlgorithmSQDSubspaceSolver
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
from cqlib_algorithms.algorithms import VarQITE as AlgorithmVarQITE
from cqlib_algorithms.algorithms import VarQITEResult as AlgorithmVarQITEResult
from cqlib_algorithms.algorithms import VarQRTE as AlgorithmVarQRTE
from cqlib_algorithms.algorithms import VarQRTEResult as AlgorithmVarQRTEResult
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
from cqlib_algorithms.algorithms import VQDResult as AlgorithmVQDResult
from cqlib_algorithms.algorithms import VQEResult as AlgorithmVQEResult
from cqlib_algorithms.algorithms import (
    recover_sqd_configurations as algorithm_recover_sqd_configurations,
)
from cqlib_algorithms.algorithms import (
    solve_sqd_subspace as algorithm_solve_sqd_subspace,
)
from cqlib_algorithms.algorithms.time_evolution import PVQD as TimeEvolutionPVQD
from cqlib_algorithms.algorithms.time_evolution import (
    PVQDResult as TimeEvolutionPVQDResult,
)
from cqlib_algorithms.algorithms.time_evolution import (
    VarQITE as TimeEvolutionVarQITE,
)
from cqlib_algorithms.algorithms.time_evolution import (
    VarQITEResult as TimeEvolutionVarQITEResult,
)
from cqlib_algorithms.algorithms.time_evolution import (
    VarQRTE as TimeEvolutionVarQRTE,
)
from cqlib_algorithms.algorithms.time_evolution import (
    VarQRTEResult as TimeEvolutionVarQRTEResult,
)
from cqlib_algorithms.algorithms.variational import QAOA as VariationalQAOA
from cqlib_algorithms.algorithms.variational import VQD as VariationalVQD
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
    VQDResult as VariationalVQDResult,
)
from cqlib_algorithms.algorithms.variational import (
    VQEResult as VariationalVQEResult,
)
from cqlib_algorithms.operators import (
    QFT as OperatorQFT,
)
from cqlib_algorithms.operators import (
    PhaseEstimationCircuit as OperatorPhaseEstimationCircuit,
)
from cqlib_algorithms.operators.composite import QFT as CompositeQFT
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
    StateFidelity,
    StatePreparation,
)

PUBLIC_COMPONENTS = {
    "AdaptVQE": AdaptVQE,
    "AdaptVQEResult": AdaptVQEResult,
    "AdaptVQETermination": AdaptVQETermination,
    "QuantumKernel": cqlib_algorithms.QuantumKernel,
    "TrainableQuantumKernel": cqlib_algorithms.TrainableQuantumKernel,
    "KernelLoss": cqlib_algorithms.KernelLoss,
    "ScalarMinimizer": cqlib_algorithms.ScalarMinimizer,
    "ScalarMinimizerResult": cqlib_algorithms.ScalarMinimizerResult,
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
    "Grover": Grover,
    "GroverRequest": GroverRequest,
    "GroverResult": GroverResult,
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
    "PVQD": PVQD,
    "PVQDResult": PVQDResult,
    "QAOA": QAOA,
    "QAOAResult": QAOAResult,
    "SQD": SQD,
    "SQDCallback": SQDCallback,
    "SQDIteration": SQDIteration,
    "SQDResult": SQDResult,
    "SQDSubspaceResult": SQDSubspaceResult,
    "SQDSubspaceSolver": SQDSubspaceSolver,
    "recover_sqd_configurations": recover_sqd_configurations,
    "solve_sqd_subspace": solve_sqd_subspace,
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
    "StateFidelity": StateFidelity,
    "StatePreparation": StatePreparation,
    "SuperdenseCoding": SuperdenseCoding,
    "SuperdenseCodingRequest": SuperdenseCodingRequest,
    "SuperdenseCodingResult": SuperdenseCodingResult,
    "StandardAmplitudeEstimation": StandardAmplitudeEstimation,
    "StandardAmplitudeEstimationRequest": StandardAmplitudeEstimationRequest,
    "StandardAmplitudeEstimationResult": StandardAmplitudeEstimationResult,
    "VQAResult": VQAResult,
    "VQD": VQD,
    "VQDResult": VQDResult,
    "VQE": VQE,
    "VQEResult": VQEResult,
    "VarQITE": VarQITE,
    "VarQITEResult": VarQITEResult,
    "VarQRTE": VarQRTE,
    "VarQRTEResult": VarQRTEResult,
    "VibrationalGroundState": VibrationalGroundState,
    "VibrationalGroundStateResult": VibrationalGroundStateResult,
    "VibrationalMeasurements": VibrationalMeasurements,
    "PegasosQSVC": PegasosQSVC,
    "QBayesian": QBayesian,
    "QBayesianResult": QBayesianResult,
    "QBayesianRound": QBayesianRound,
    "QBayesianStatus": QBayesianStatus,
    "QSVR": QSVR,
    "QuantumKernelTrainer": QuantumKernelTrainer,
    "QuantumKernelTrainerResult": QuantumKernelTrainerResult,
    "VQR": VQR,
}


def test_distribution_metadata():
    distribution = metadata("cqlib-algorithms")
    assert distribution["Requires-Python"] == ">=3.10"
    assert cqlib_algorithms.__version__ == distribution["Version"]


def test_algorithms_package_is_importable():
    assert import_module("cqlib_algorithms.algorithms") is not None


def test_protocols_package_is_importable():
    assert import_module("cqlib_algorithms.protocols") is not None


def test_composite_operators_package_is_importable():
    assert import_module("cqlib_algorithms.operators.composite") is not None


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
        StateFidelity,
        StatePreparation,
    ):
        assert inspect.isabstract(component)


def test_qft_is_exported_from_public_namespaces():
    assert QFT is OperatorQFT is CompositeQFT


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
    assert PVQD is AlgorithmPVQD is TimeEvolutionPVQD
    assert PVQDResult is AlgorithmPVQDResult is TimeEvolutionPVQDResult
    assert VQAResult is AlgorithmVQAResult is VariationalVQAResult


def test_var_qite_is_exported_from_public_namespaces():
    assert VarQITE is AlgorithmVarQITE is TimeEvolutionVarQITE
    assert VarQITEResult is AlgorithmVarQITEResult is TimeEvolutionVarQITEResult


def test_var_qrte_is_exported_from_public_namespaces():
    assert VarQRTE is AlgorithmVarQRTE is TimeEvolutionVarQRTE
    assert VarQRTEResult is AlgorithmVarQRTEResult is TimeEvolutionVarQRTEResult


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


def test_vqd_is_exported_from_public_namespaces():
    assert VQD is AlgorithmVQD is VariationalVQD
    assert VQDResult is AlgorithmVQDResult is VariationalVQDResult


def test_electronic_ground_state_is_exported_from_public_namespaces():
    assert ElectronicGroundState is AlgorithmElectronicGroundState
    assert ElectronicGroundStateResult is AlgorithmElectronicGroundStateResult
    assert ElectronicQEOM is AlgorithmElectronicQEOM
    assert ElectronicQEOMResult is AlgorithmElectronicQEOMResult


def test_sqd_is_exported_from_public_namespaces():
    assert SQD is AlgorithmSQD
    assert SQDCallback is AlgorithmSQDCallback
    assert SQDIteration is AlgorithmSQDIteration
    assert SQDResult is AlgorithmSQDResult
    assert SQDSubspaceResult is AlgorithmSQDSubspaceResult
    assert SQDSubspaceSolver is AlgorithmSQDSubspaceSolver
    assert recover_sqd_configurations is algorithm_recover_sqd_configurations
    assert solve_sqd_subspace is algorithm_solve_sqd_subspace


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


def test_grover_is_exported_from_public_namespaces():
    assert Grover is AlgorithmGrover
    assert GroverRequest is AlgorithmGroverRequest
    assert GroverResult is AlgorithmGroverResult


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


@pytest.mark.parametrize(
    "name",
    [
        "PegasosQSVC",
        "QBayesian",
        "QBayesianResult",
        "QBayesianRound",
        "QBayesianStatus",
        "QSVR",
        "QuantumKernelTrainer",
        "QuantumKernelTrainerResult",
        "VQR",
    ],
)
def test_qml_algorithms_are_exported_from_public_namespaces(name):
    exported = getattr(cqlib_algorithms, name)
    assert exported is getattr(import_module("cqlib_algorithms.algorithms"), name)
    assert exported is getattr(
        import_module("cqlib_algorithms.algorithms.machine_learning"), name
    )
