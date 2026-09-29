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

from .bb84 import BB84, BB84Basis, BB84Request, BB84Result
from .bernstein_vazirani import (
    BernsteinVazirani,
    BernsteinVaziraniRequest,
    BernsteinVaziraniResult,
)
from .chsh import CHSH, CHSHRequest, CHSHResult
from .deutsch_jozsa import DeutschJozsa, DeutschJozsaRequest, DeutschJozsaResult
from .qaoa import QAOA, QAOAResult
from .quantum_teleportation import (
    QuantumTeleportation,
    QuantumTeleportationRequest,
    QuantumTeleportationResult,
)
from .simon import Simon, SimonRequest, SimonResult
from .vqa_result import VQAResult
from .vqe import VQE, VQEResult

__all__ = [
    "BB84",
    "CHSH",
    "QAOA",
    "VQE",
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
    "QAOAResult",
    "QuantumTeleportation",
    "QuantumTeleportationRequest",
    "QuantumTeleportationResult",
    "Simon",
    "SimonRequest",
    "SimonResult",
    "VQAResult",
    "VQEResult",
]
