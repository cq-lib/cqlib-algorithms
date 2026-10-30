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

r"""Quasi-probability decompositions for error mitigation.

Both error mitigation families that trade circuit executions for accuracy
build on this module. PEC inverts the device noise into signed Pauli
corrections of 1-norm :math:`\Gamma > 1`, while PEA amplifies it into a
genuine probability distribution of 1-norm one. Neither ever constructs a
noisy gate: the noise is a premise, the Pauli corrections are the only
executable gates.
"""

from .amplifications import amplify_depolarizing
from .depolarizing import (
    QuasiprobabilityRepresentation,
    depolarizing_quasiprobability_of_circuit,
    depolarizing_quasiprobability_of_operation,
    quasiprobability_of_circuit,
    quasiprobability_of_operation,
)

__all__ = [
    "QuasiprobabilityRepresentation",
    "amplify_depolarizing",
    "depolarizing_quasiprobability_of_circuit",
    "depolarizing_quasiprobability_of_operation",
    "quasiprobability_of_circuit",
    "quasiprobability_of_operation",
]
