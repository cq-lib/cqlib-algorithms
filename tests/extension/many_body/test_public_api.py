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

"""Public API checks for temporary many-body representations."""

from importlib import import_module


def test_many_body_public_api_is_explicit():
    many_body = import_module("cqlib_algorithms.extension.many_body")
    expected = {
        "ModalSpace",
        "OneHotVibrationalMapper",
        "SpinAxis",
        "SpinFactor",
        "SpinHalfMapper",
        "SpinOp",
        "SpinTerm",
        "VibrationalCoefficient",
        "VibrationalLadder",
        "VibrationalLadderKind",
        "VibrationalOp",
        "VibrationalTerm",
    }

    assert set(many_body.__all__) == expected
    for name in expected:
        assert getattr(many_body, name) is not None
