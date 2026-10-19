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
