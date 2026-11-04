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

"""Dependency-direction tests for the private many-body mapping kernels."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_SOURCE_DIRECTORY = (
    Path(__file__).parents[3] / "src" / "cqlib_algorithms" / "extension" / "many_body"
)


def _imported_modules(filename: str) -> set[str]:
    tree = ast.parse((_SOURCE_DIRECTORY / filename).read_text(encoding="utf-8"))
    return {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }


def test_operator_representations_do_not_import_public_mappers() -> None:
    assert "spin_half_mapping" not in _imported_modules("spin_operator.py")
    assert "one_hot_vibrational_mapping" not in _imported_modules(
        "vibrational_operator.py"
    )


def test_private_mapping_kernel_does_not_import_representations_or_mappers() -> None:
    imports = _imported_modules("_qubit_mapping.py")
    assert imports.isdisjoint(
        {
            "spin_operator",
            "vibrational_operator",
            "spin_half_mapping",
            "one_hot_vibrational_mapping",
        }
    )


@pytest.mark.parametrize(
    "module_order",
    [
        (
            "spin_operator",
            "spin_half_mapping",
            "vibrational_operator",
            "one_hot_vibrational_mapping",
        ),
        (
            "one_hot_vibrational_mapping",
            "vibrational_operator",
            "spin_half_mapping",
            "spin_operator",
        ),
    ],
)
def test_many_body_modules_support_different_import_orders(
    module_order: tuple[str, ...],
) -> None:
    imports = "\n".join(
        f"importlib.import_module('cqlib_algorithms.extension.many_body.{module_name}')"
        for module_name in module_order
    )
    source = f"""
import importlib
{imports}

from cqlib_algorithms.extension.many_body import (
    ModalSpace,
    OneHotVibrationalMapper,
    SpinHalfMapper,
    SpinOp,
    VibrationalLadder,
    VibrationalOp,
)

spin = SpinOp.from_terms(1, (((('X', 0),), 1.0),))
assert SpinHalfMapper().map_operator(spin).num_qubits == 1
space = ModalSpace((2,))
ladder = VibrationalLadder('+', 0, 1)
vibration = VibrationalOp.from_terms(space, (((ladder,), 1.0),))
assert OneHotVibrationalMapper().map_operator(vibration).num_qubits == 2
"""
    subprocess.run([sys.executable, "-c", source], check=True)
