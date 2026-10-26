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

r"""Noise scaling primitives for error mitigation.

Every function here maps one circuit to one higher-noise circuit while
leaving the logical action unchanged, which is what makes the extrapolation
in :mod:`cqlib_algorithms.error_mitigation.inference` meaningful.
"""

from .folding import fold_gates_at_random, fold_global, fold_local
from .identity_insertion import insert_id_layers
from .layer_scaling import fold_layers
from .parameter import scale_parameters

__all__ = [
    "fold_gates_at_random",
    "fold_global",
    "fold_layers",
    "fold_local",
    "insert_id_layers",
    "scale_parameters",
]
