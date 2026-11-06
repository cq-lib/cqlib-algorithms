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

"""Clean-process tests for the domain-extension facade."""

from __future__ import annotations

import json
import subprocess
import sys


def _run_clean_process(source: str) -> object:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_extension_import_does_not_eagerly_load_domain_namespaces() -> None:
    result = _run_clean_process(
        """
import json
import sys

import cqlib_algorithms.extension

domains = [
    "cqlib_algorithms.extension.electronic_structure",
    "cqlib_algorithms.extension.lattice",
    "cqlib_algorithms.extension.many_body",
    "cqlib_algorithms.extension.vibration",
]
print(json.dumps([name in sys.modules for name in domains]))
"""
    )

    assert result == [False, False, False, False]


def test_extension_namespaces_aliases_and_import_protocol_are_compatible() -> None:
    result = _run_clean_process(
        """
import importlib
import json

import cqlib_algorithms.extension as extension
from cqlib_algorithms.extension import (
    electronic_structure,
    lattice,
    many_body,
    vibration,
)

domain_modules = [electronic_structure, lattice, many_body, vibration]
domain_names = ["electronic_structure", "lattice", "many_body", "vibration"]
namespace = {}
exec("from cqlib_algorithms.extension import *", namespace)

unknown_raises = False
try:
    extension.not_a_public_extension
except AttributeError:
    unknown_raises = True

print(
    json.dumps(
        {
            "aliases": [
                extension.ActiveSpace is electronic_structure.ActiveSpace,
                extension.BogoliubovTransform
                is electronic_structure.BogoliubovTransform,
                extension.transform_active_space
                is electronic_structure.transform_active_space,
            ],
            "cached": [
                extension.__dict__[name] is module
                for name, module in zip(domain_names, domain_modules)
            ],
            "dir": set(extension.__all__).issubset(dir(extension)),
            "modules": [
                module is importlib.import_module(
                    f"cqlib_algorithms.extension.{name}"
                )
                for name, module in zip(domain_names, domain_modules)
            ],
            "star": sorted(name for name in namespace if name != "__builtins__")
            == sorted(extension.__all__),
            "unknown": unknown_raises,
        }
    )
)
"""
    )

    assert result == {
        "aliases": [True, True, True],
        "cached": [True, True, True, True],
        "dir": True,
        "modules": [True, True, True, True],
        "star": True,
        "unknown": True,
    }
