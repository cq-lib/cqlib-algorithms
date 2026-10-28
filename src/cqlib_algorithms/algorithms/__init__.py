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
    ElectronicGroundState,
    ElectronicGroundStateResult,
    ElectronicQEOM,
    ElectronicQEOMResult,
)
from .lattice import LatticeGroundState, LatticeGroundStateResult
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
from .time_evolution import PVQD, PVQDResult
from .variational import (
    QAOA,
    VQE,
    AdaptVQE,
    AdaptVQEResult,
    AdaptVQETermination,
    QAOAResult,
    SamplingVQE,
    SamplingVQEMeasurement,
    SamplingVQEResult,
    VQAResult,
    VQEResult,
)
from .vibration import (
    VibrationalGroundState,
    VibrationalGroundStateResult,
    VibrationalMeasurements,
)

__all__ = [
    "BB84",
    "CHSH",
    "PVQD",
    "QAOA",
    "VQE",
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
    "PhaseEstimation",
    "PhaseEstimationProblem",
    "PhaseEstimationRequest",
    "PhaseEstimationResult",
    "QAOAResult",
    "QuantumTeleportation",
    "QuantumTeleportationRequest",
    "QuantumTeleportationResult",
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
    "VQEResult",
    "VibrationalGroundState",
    "VibrationalGroundStateResult",
    "VibrationalMeasurements",
]
