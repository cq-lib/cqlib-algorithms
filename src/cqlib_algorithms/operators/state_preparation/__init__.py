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

"""State-preparation and data-encoding components."""

from .amplitude_encoding import AmplitudeEncoding
from .angle_encoding import AngleEncoding
from .arbitrary_state_preparation import ArbitraryStatePreparation
from .basis_encoding import BasisEncoding
from .basis_state_initialization import BasisStateInitialization
from .dicke_state_preparation import DickeStatePreparation
from .feature_map_preparation import FeatureMapPreparation
from .ghz_state_preparation import GHZStatePreparation
from .product_state_preparation import ProductStatePreparation
from .sparse_amplitude_encoding import SparseAmplitudeEncoding
from .uniform_superposition import UniformSuperposition
from .user_circuit_preparation import UserCircuitPreparation

__all__ = [
    "AmplitudeEncoding",
    "AngleEncoding",
    "ArbitraryStatePreparation",
    "BasisEncoding",
    "BasisStateInitialization",
    "DickeStatePreparation",
    "FeatureMapPreparation",
    "GHZStatePreparation",
    "ProductStatePreparation",
    "SparseAmplitudeEncoding",
    "UniformSuperposition",
    "UserCircuitPreparation",
]
