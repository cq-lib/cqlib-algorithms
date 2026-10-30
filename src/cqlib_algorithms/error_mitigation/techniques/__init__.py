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

r"""Error-mitigation techniques.

Each technique provides an immediate ``execute_with_<m>`` helper and a
``<m>_executor`` factory. The factory returns a
:class:`~cqlib_algorithms.error_mitigation.core.executor.QemExecutor` whose
estimator backend applies the technique, so techniques compose by nesting
their factories.

Dynamical decoupling and Pauli twirling are line transforms first, so their
:class:`DynamicalDecoupling` and :class:`PauliTwirling` objects can also be
applied to a circuit directly. Dynamical decoupling additionally exports the
rules it lays into idle windows.
"""

from ._ddd_rules import XX, XYXY, YY, DDDRule, general_rule
from .cdr import cdr_executor, execute_with_cdr
from .ddd import DynamicalDecoupling, ddd_executor, execute_with_ddd
from .deb import deb_executor, execute_with_deb
from .lre import execute_with_lre, lre_executor
from .pea import execute_with_pea, pea_executor
from .pec import execute_with_pec, pec_executor
from .pt import PauliTwirling, execute_with_pt, pt_executor
from .qse import execute_with_qse, qse_executor
from .rem import (
    ConfusionMatrix,
    calibrate_readout,
    execute_with_rem,
    mitigate_counts,
    rem_executor,
)
from .shadows import execute_with_shadows, shadows_executor
from .trex import execute_with_trex, trex_executor
from .vd import execute_with_vd, vd_executor
from .zne import ScalingMethod, execute_with_zne, zne_executor

__all__ = [
    "XX",
    "XYXY",
    "YY",
    "ConfusionMatrix",
    "DDDRule",
    "DynamicalDecoupling",
    "PauliTwirling",
    "ScalingMethod",
    "calibrate_readout",
    "cdr_executor",
    "ddd_executor",
    "deb_executor",
    "execute_with_cdr",
    "execute_with_ddd",
    "execute_with_deb",
    "execute_with_lre",
    "execute_with_pea",
    "execute_with_pec",
    "execute_with_pt",
    "execute_with_qse",
    "execute_with_rem",
    "execute_with_shadows",
    "execute_with_trex",
    "execute_with_vd",
    "execute_with_zne",
    "general_rule",
    "lre_executor",
    "mitigate_counts",
    "pea_executor",
    "pec_executor",
    "pt_executor",
    "qse_executor",
    "rem_executor",
    "shadows_executor",
    "trex_executor",
    "vd_executor",
    "zne_executor",
]
