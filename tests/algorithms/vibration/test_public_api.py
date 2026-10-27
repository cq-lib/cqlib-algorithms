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

"""Public namespace tests for vibrational algorithms."""

from cqlib_algorithms import (
    VibrationalGroundState,
    VibrationalGroundStateResult,
    VibrationalMeasurements,
)
from cqlib_algorithms.algorithms import (
    VibrationalGroundState as AlgorithmVibrationalGroundState,
)
from cqlib_algorithms.algorithms import (
    VibrationalGroundStateResult as AlgorithmVibrationalGroundStateResult,
)
from cqlib_algorithms.algorithms import (
    VibrationalMeasurements as AlgorithmVibrationalMeasurements,
)
from cqlib_algorithms.algorithms.vibration import (
    VibrationalGroundState as DomainVibrationalGroundState,
)
from cqlib_algorithms.algorithms.vibration import (
    VibrationalGroundStateResult as DomainVibrationalGroundStateResult,
)
from cqlib_algorithms.algorithms.vibration import (
    VibrationalMeasurements as DomainVibrationalMeasurements,
)


def test_vibrational_algorithm_is_exported_from_public_namespaces():
    assert (
        VibrationalGroundState
        is AlgorithmVibrationalGroundState
        is DomainVibrationalGroundState
    )
    assert (
        VibrationalGroundStateResult
        is AlgorithmVibrationalGroundStateResult
        is DomainVibrationalGroundStateResult
    )
    assert (
        VibrationalMeasurements
        is AlgorithmVibrationalMeasurements
        is DomainVibrationalMeasurements
    )
