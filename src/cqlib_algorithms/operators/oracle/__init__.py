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

"""Boolean oracle and reflection circuit components."""

from .bit_flip_oracle import BitFlipOracle
from .block_encoding import BlockEncoding, UnitaryDilation
from .boolean_function import BooleanFunction
from .boolean_vector_function import BooleanVectorFunction
from .dicke_unitary import DickeUnitary
from .general_state_reflection import GeneralStateReflection
from .grover_diffusion import GroverDiffusion
from .grover_long import GroverLong
from .phase_oracle import PhaseOracle
from .quantum_singular_value_transformation import (
    ProjectorControlledPhase,
    QuantumSignalProcessing,
    QuantumSingularValueTransformation,
)
from .split_cycle_shift import SplitCycleShift
from .unitary_matrix_oracle import UnitaryMatrixOracle
from .weight_distribution_block import WeightDistributionBlock
from .zero_state_reflection import ZeroStateReflection

__all__ = [
    "BitFlipOracle",
    "BlockEncoding",
    "BooleanFunction",
    "BooleanVectorFunction",
    "DickeUnitary",
    "GeneralStateReflection",
    "GroverDiffusion",
    "GroverLong",
    "PhaseOracle",
    "ProjectorControlledPhase",
    "QuantumSignalProcessing",
    "QuantumSingularValueTransformation",
    "SplitCycleShift",
    "UnitaryDilation",
    "UnitaryMatrixOracle",
    "WeightDistributionBlock",
    "ZeroStateReflection",
]
