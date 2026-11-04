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

"""Quantum machine-learning algorithms and training workflows."""

from .pegasos_qsvc import PegasosQSVC
from .qbayesian import QBayesian, QBayesianResult, QBayesianRound, QBayesianStatus
from .qsvr import QSVR
from .quantum_kernel_trainer import QuantumKernelTrainer, QuantumKernelTrainerResult
from .vqr import VQR

__all__ = [
    "QSVR",
    "VQR",
    "PegasosQSVC",
    "QBayesian",
    "QBayesianResult",
    "QBayesianRound",
    "QBayesianStatus",
    "QuantumKernelTrainer",
    "QuantumKernelTrainerResult",
]
