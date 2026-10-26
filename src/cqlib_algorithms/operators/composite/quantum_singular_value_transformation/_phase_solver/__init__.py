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

r"""Classical phase-synthesis routines for quantum signal processing.

This subpackage groups the purely numerical pre-processing steps that turn a
polynomial specification into a QSP / QSVT phase sequence.  It is intentionally
decoupled from the quantum circuit construction in :mod:`qsp` and
:mod:`qsvt`.

Public API
----------
- :func:`solve_qsp_phases` — real Chebyshev polynomial → QSP phases.
- :func:`solve_complex_qsp_phases` — complex Chebyshev polynomial → QSP phases.
- :func:`solve_haah_phases` — complex Chebyshev polynomial → QSP phases, via
  Haah's deterministic product decomposition.
- :func:`polynomial_to_chebyshev` — power-basis → Chebyshev-basis conversion.
- :func:`convert_qsp_phases` — full symmetric QSP phases → QSVT ``W``-form.
"""

from ._complex_solver import solve_complex_qsp_phases
from ._converters import convert_qsp_phases, polynomial_to_chebyshev
from ._haah_solver import solve_haah_phases
from ._real_solver import solve_qsp_phases

__all__ = [
    "convert_qsp_phases",
    "polynomial_to_chebyshev",
    "solve_complex_qsp_phases",
    "solve_haah_phases",
    "solve_qsp_phases",
]
