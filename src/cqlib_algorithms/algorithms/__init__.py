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

"""Complete quantum algorithms composed from reusable components."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .amplitude_estimation import (
        AmplitudeEstimationProblem,
        FasterAmplitudeEstimation,
        FasterAmplitudeEstimationRequest,
        FasterAmplitudeEstimationResult,
        IterativeAmplitudeEstimation,
        IterativeAmplitudeEstimationRequest,
        IterativeAmplitudeEstimationResult,
        MaximumLikelihoodAmplitudeEstimation,
        MaximumLikelihoodAmplitudeEstimationRequest,
        MaximumLikelihoodAmplitudeEstimationResult,
        StandardAmplitudeEstimation,
        StandardAmplitudeEstimationRequest,
        StandardAmplitudeEstimationResult,
    )
    from .bb84 import BB84, BB84Basis, BB84Request, BB84Result
    from .bernstein_vazirani import (
        BernsteinVazirani,
        BernsteinVaziraniRequest,
        BernsteinVaziraniResult,
    )
    from .chsh import CHSH, CHSHRequest, CHSHResult
    from .deutsch_jozsa import DeutschJozsa, DeutschJozsaRequest, DeutschJozsaResult
    from .electronic_structure import (
        SQD,
        ElectronicGroundState,
        ElectronicGroundStateResult,
        ElectronicQEOM,
        ElectronicQEOMResult,
        SQDCallback,
        SQDIteration,
        SQDResult,
        SQDSubspaceResult,
        SQDSubspaceSolver,
        recover_sqd_configurations,
        solve_sqd_subspace,
    )
    from .grover import Grover, GroverRequest, GroverResult
    from .lattice import LatticeGroundState, LatticeGroundStateResult
    from .machine_learning import (
        QSVR,
        VQR,
        PegasosQSVC,
        QBayesian,
        QBayesianResult,
        QBayesianRound,
        QBayesianStatus,
        QuantumKernelTrainer,
        QuantumKernelTrainerResult,
    )
    from .phase_estimation import (
        HamiltonianPhaseEstimation,
        HamiltonianPhaseEstimationRequest,
        HamiltonianPhaseEstimationResult,
        IterativePhaseEstimation,
        IterativePhaseEstimationRequest,
        IterativePhaseEstimationResult,
        PhaseEstimation,
        PhaseEstimationProblem,
        PhaseEstimationRequest,
        PhaseEstimationResult,
    )
    from .quantum_teleportation import (
        QuantumTeleportation,
        QuantumTeleportationRequest,
        QuantumTeleportationResult,
    )
    from .simon import Simon, SimonRequest, SimonResult
    from .superdense_coding import (
        SuperdenseCoding,
        SuperdenseCodingRequest,
        SuperdenseCodingResult,
    )
    from .time_evolution import (
        PVQD,
        PVQDResult,
        VarQITE,
        VarQITEResult,
        VarQRTE,
        VarQRTEResult,
    )
    from .variational import (
        QAOA,
        VQD,
        VQE,
        AdaptVQE,
        AdaptVQEResult,
        AdaptVQETermination,
        QAOAResult,
        SamplingVQE,
        SamplingVQEMeasurement,
        SamplingVQEResult,
        VQAResult,
        VQDResult,
        VQEResult,
    )
    from .vibration import (
        VibrationalGroundState,
        VibrationalGroundStateResult,
        VibrationalMeasurements,
    )

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "BB84": (".bb84", "BB84"),
    "CHSH": (".chsh", "CHSH"),
    "PVQD": (".time_evolution", "PVQD"),
    "QAOA": (".variational", "QAOA"),
    "QSVR": (".machine_learning", "QSVR"),
    "SQD": (".electronic_structure", "SQD"),
    "VQD": (".variational", "VQD"),
    "VQE": (".variational", "VQE"),
    "VQR": (".machine_learning", "VQR"),
    "AdaptVQE": (".variational", "AdaptVQE"),
    "AdaptVQEResult": (".variational", "AdaptVQEResult"),
    "AdaptVQETermination": (".variational", "AdaptVQETermination"),
    "AmplitudeEstimationProblem": (
        ".amplitude_estimation",
        "AmplitudeEstimationProblem",
    ),
    "BB84Basis": (".bb84", "BB84Basis"),
    "BB84Request": (".bb84", "BB84Request"),
    "BB84Result": (".bb84", "BB84Result"),
    "BernsteinVazirani": (".bernstein_vazirani", "BernsteinVazirani"),
    "BernsteinVaziraniRequest": (
        ".bernstein_vazirani",
        "BernsteinVaziraniRequest",
    ),
    "BernsteinVaziraniResult": (
        ".bernstein_vazirani",
        "BernsteinVaziraniResult",
    ),
    "CHSHRequest": (".chsh", "CHSHRequest"),
    "CHSHResult": (".chsh", "CHSHResult"),
    "DeutschJozsa": (".deutsch_jozsa", "DeutschJozsa"),
    "DeutschJozsaRequest": (".deutsch_jozsa", "DeutschJozsaRequest"),
    "DeutschJozsaResult": (".deutsch_jozsa", "DeutschJozsaResult"),
    "ElectronicGroundState": (".electronic_structure", "ElectronicGroundState"),
    "ElectronicGroundStateResult": (
        ".electronic_structure",
        "ElectronicGroundStateResult",
    ),
    "ElectronicQEOM": (".electronic_structure", "ElectronicQEOM"),
    "ElectronicQEOMResult": (".electronic_structure", "ElectronicQEOMResult"),
    "FasterAmplitudeEstimation": (
        ".amplitude_estimation",
        "FasterAmplitudeEstimation",
    ),
    "FasterAmplitudeEstimationRequest": (
        ".amplitude_estimation",
        "FasterAmplitudeEstimationRequest",
    ),
    "FasterAmplitudeEstimationResult": (
        ".amplitude_estimation",
        "FasterAmplitudeEstimationResult",
    ),
    "Grover": (".grover", "Grover"),
    "GroverRequest": (".grover", "GroverRequest"),
    "GroverResult": (".grover", "GroverResult"),
    "HamiltonianPhaseEstimation": (
        ".phase_estimation",
        "HamiltonianPhaseEstimation",
    ),
    "HamiltonianPhaseEstimationRequest": (
        ".phase_estimation",
        "HamiltonianPhaseEstimationRequest",
    ),
    "HamiltonianPhaseEstimationResult": (
        ".phase_estimation",
        "HamiltonianPhaseEstimationResult",
    ),
    "IterativeAmplitudeEstimation": (
        ".amplitude_estimation",
        "IterativeAmplitudeEstimation",
    ),
    "IterativeAmplitudeEstimationRequest": (
        ".amplitude_estimation",
        "IterativeAmplitudeEstimationRequest",
    ),
    "IterativeAmplitudeEstimationResult": (
        ".amplitude_estimation",
        "IterativeAmplitudeEstimationResult",
    ),
    "IterativePhaseEstimation": (
        ".phase_estimation",
        "IterativePhaseEstimation",
    ),
    "IterativePhaseEstimationRequest": (
        ".phase_estimation",
        "IterativePhaseEstimationRequest",
    ),
    "IterativePhaseEstimationResult": (
        ".phase_estimation",
        "IterativePhaseEstimationResult",
    ),
    "LatticeGroundState": (".lattice", "LatticeGroundState"),
    "LatticeGroundStateResult": (".lattice", "LatticeGroundStateResult"),
    "MaximumLikelihoodAmplitudeEstimation": (
        ".amplitude_estimation",
        "MaximumLikelihoodAmplitudeEstimation",
    ),
    "MaximumLikelihoodAmplitudeEstimationRequest": (
        ".amplitude_estimation",
        "MaximumLikelihoodAmplitudeEstimationRequest",
    ),
    "MaximumLikelihoodAmplitudeEstimationResult": (
        ".amplitude_estimation",
        "MaximumLikelihoodAmplitudeEstimationResult",
    ),
    "PVQDResult": (".time_evolution", "PVQDResult"),
    "PegasosQSVC": (".machine_learning", "PegasosQSVC"),
    "PhaseEstimation": (".phase_estimation", "PhaseEstimation"),
    "PhaseEstimationProblem": (".phase_estimation", "PhaseEstimationProblem"),
    "PhaseEstimationRequest": (".phase_estimation", "PhaseEstimationRequest"),
    "PhaseEstimationResult": (".phase_estimation", "PhaseEstimationResult"),
    "QAOAResult": (".variational", "QAOAResult"),
    "QBayesian": (".machine_learning", "QBayesian"),
    "QBayesianResult": (".machine_learning", "QBayesianResult"),
    "QBayesianRound": (".machine_learning", "QBayesianRound"),
    "QBayesianStatus": (".machine_learning", "QBayesianStatus"),
    "QuantumKernelTrainer": (".machine_learning", "QuantumKernelTrainer"),
    "QuantumKernelTrainerResult": (
        ".machine_learning",
        "QuantumKernelTrainerResult",
    ),
    "QuantumTeleportation": (".quantum_teleportation", "QuantumTeleportation"),
    "QuantumTeleportationRequest": (
        ".quantum_teleportation",
        "QuantumTeleportationRequest",
    ),
    "QuantumTeleportationResult": (
        ".quantum_teleportation",
        "QuantumTeleportationResult",
    ),
    "SQDCallback": (".electronic_structure", "SQDCallback"),
    "SQDIteration": (".electronic_structure", "SQDIteration"),
    "SQDResult": (".electronic_structure", "SQDResult"),
    "SQDSubspaceResult": (".electronic_structure", "SQDSubspaceResult"),
    "SQDSubspaceSolver": (".electronic_structure", "SQDSubspaceSolver"),
    "SamplingVQE": (".variational", "SamplingVQE"),
    "SamplingVQEMeasurement": (".variational", "SamplingVQEMeasurement"),
    "SamplingVQEResult": (".variational", "SamplingVQEResult"),
    "Simon": (".simon", "Simon"),
    "SimonRequest": (".simon", "SimonRequest"),
    "SimonResult": (".simon", "SimonResult"),
    "StandardAmplitudeEstimation": (
        ".amplitude_estimation",
        "StandardAmplitudeEstimation",
    ),
    "StandardAmplitudeEstimationRequest": (
        ".amplitude_estimation",
        "StandardAmplitudeEstimationRequest",
    ),
    "StandardAmplitudeEstimationResult": (
        ".amplitude_estimation",
        "StandardAmplitudeEstimationResult",
    ),
    "SuperdenseCoding": (".superdense_coding", "SuperdenseCoding"),
    "SuperdenseCodingRequest": (
        ".superdense_coding",
        "SuperdenseCodingRequest",
    ),
    "SuperdenseCodingResult": (
        ".superdense_coding",
        "SuperdenseCodingResult",
    ),
    "VQAResult": (".variational", "VQAResult"),
    "VQDResult": (".variational", "VQDResult"),
    "VQEResult": (".variational", "VQEResult"),
    "VarQITE": (".time_evolution", "VarQITE"),
    "VarQITEResult": (".time_evolution", "VarQITEResult"),
    "VarQRTE": (".time_evolution", "VarQRTE"),
    "VarQRTEResult": (".time_evolution", "VarQRTEResult"),
    "VibrationalGroundState": (".vibration", "VibrationalGroundState"),
    "VibrationalGroundStateResult": (
        ".vibration",
        "VibrationalGroundStateResult",
    ),
    "VibrationalMeasurements": (".vibration", "VibrationalMeasurements"),
    "recover_sqd_configurations": (
        ".electronic_structure",
        "recover_sqd_configurations",
    ),
    "solve_sqd_subspace": (".electronic_structure", "solve_sqd_subspace"),
}

__all__ = [
    "BB84",
    "CHSH",
    "PVQD",
    "QAOA",
    "QSVR",
    "SQD",
    "VQD",
    "VQE",
    "VQR",
    "AdaptVQE",
    "AdaptVQEResult",
    "AdaptVQETermination",
    "AmplitudeEstimationProblem",
    "BB84Basis",
    "BB84Request",
    "BB84Result",
    "BernsteinVazirani",
    "BernsteinVaziraniRequest",
    "BernsteinVaziraniResult",
    "CHSHRequest",
    "CHSHResult",
    "DeutschJozsa",
    "DeutschJozsaRequest",
    "DeutschJozsaResult",
    "ElectronicGroundState",
    "ElectronicGroundStateResult",
    "ElectronicQEOM",
    "ElectronicQEOMResult",
    "FasterAmplitudeEstimation",
    "FasterAmplitudeEstimationRequest",
    "FasterAmplitudeEstimationResult",
    "Grover",
    "GroverRequest",
    "GroverResult",
    "HamiltonianPhaseEstimation",
    "HamiltonianPhaseEstimationRequest",
    "HamiltonianPhaseEstimationResult",
    "IterativeAmplitudeEstimation",
    "IterativeAmplitudeEstimationRequest",
    "IterativeAmplitudeEstimationResult",
    "IterativePhaseEstimation",
    "IterativePhaseEstimationRequest",
    "IterativePhaseEstimationResult",
    "LatticeGroundState",
    "LatticeGroundStateResult",
    "MaximumLikelihoodAmplitudeEstimation",
    "MaximumLikelihoodAmplitudeEstimationRequest",
    "MaximumLikelihoodAmplitudeEstimationResult",
    "PVQDResult",
    "PegasosQSVC",
    "PhaseEstimation",
    "PhaseEstimationProblem",
    "PhaseEstimationRequest",
    "PhaseEstimationResult",
    "QAOAResult",
    "QBayesian",
    "QBayesianResult",
    "QBayesianRound",
    "QBayesianStatus",
    "QuantumKernelTrainer",
    "QuantumKernelTrainerResult",
    "QuantumTeleportation",
    "QuantumTeleportationRequest",
    "QuantumTeleportationResult",
    "SQDCallback",
    "SQDIteration",
    "SQDResult",
    "SQDSubspaceResult",
    "SQDSubspaceSolver",
    "SamplingVQE",
    "SamplingVQEMeasurement",
    "SamplingVQEResult",
    "Simon",
    "SimonRequest",
    "SimonResult",
    "StandardAmplitudeEstimation",
    "StandardAmplitudeEstimationRequest",
    "StandardAmplitudeEstimationResult",
    "SuperdenseCoding",
    "SuperdenseCodingRequest",
    "SuperdenseCodingResult",
    "VQAResult",
    "VQDResult",
    "VQEResult",
    "VarQITE",
    "VarQITEResult",
    "VarQRTE",
    "VarQRTEResult",
    "VibrationalGroundState",
    "VibrationalGroundStateResult",
    "VibrationalMeasurements",
    "recover_sqd_configurations",
    "solve_sqd_subspace",
]


def __getattr__(name: str) -> object:
    """Resolve and cache a public export on first access."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value: object = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals together with lazily exported names."""
    return sorted(set(globals()) | set(__all__))
