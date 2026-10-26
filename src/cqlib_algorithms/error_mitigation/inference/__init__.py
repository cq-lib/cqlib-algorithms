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

r"""Classical inference for error mitigation.

Both halves of the branch share this module: ZNE-style methods measure one
expectation value per scalar scale factor and extrapolate to zero noise,
while LRE measures one per scale vector and combines them with the
multivariate weights. Every fit is implemented with NumPy only.
"""

from .factories import (
    AdaExpFactory,
    ConvergenceWarning,
    ExpFactory,
    ExtrapolationError,
    Factory,
    LinearFactory,
    PolyExpFactory,
    PolyFactory,
    RichardsonFactory,
)
from .multivariate import multivariate_richardson_coefficients

__all__ = [
    "AdaExpFactory",
    "ConvergenceWarning",
    "ExpFactory",
    "ExtrapolationError",
    "Factory",
    "LinearFactory",
    "PolyExpFactory",
    "PolyFactory",
    "RichardsonFactory",
    "multivariate_richardson_coefficients",
]
