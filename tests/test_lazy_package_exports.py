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

"""Clean-process tests for the public package facades."""

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


def test_root_import_does_not_eagerly_load_scipy_or_internal_packages() -> None:
    result = _run_clean_process(
        """
import json
import sys

import cqlib_algorithms

internal_modules = [
    name
    for name in sys.modules
    if name == "cqlib_algorithms" or name.startswith("cqlib_algorithms.")
]
print(json.dumps({"modules": internal_modules, "scipy": "scipy" in sys.modules}))
"""
    )

    assert isinstance(result, dict)
    assert result["scipy"] is False
    assert result["modules"] == ["cqlib_algorithms"]


def test_all_public_names_resolve_cache_and_appear_in_dir() -> None:
    result = _run_clean_process(
        """
import importlib
import json

facades = [
    importlib.import_module("cqlib_algorithms"),
    importlib.import_module("cqlib_algorithms.algorithms"),
    importlib.import_module("cqlib_algorithms.operators"),
]
resolved = []
for facade in facades:
    values = [getattr(facade, name) for name in facade.__all__]
    resolved.append(
        all(name in facade.__dict__ for name in facade.__all__)
        and set(facade.__all__).issubset(dir(facade))
        and all(value is getattr(facade, name) for name, value in zip(facade.__all__, values))
    )
print(json.dumps(resolved))
"""
    )

    assert result == [True, True, True]


def test_from_and_star_imports_preserve_public_exports() -> None:
    result = _run_clean_process(
        """
import json
from types import ModuleType

import cqlib_algorithms
import cqlib_algorithms.algorithms as algorithms
import cqlib_algorithms.operators as operators
from cqlib_algorithms import VQE
from cqlib_algorithms.operators import arithmetic, mutual_information

namespaces = []
for module_name in (
    "cqlib_algorithms",
    "cqlib_algorithms.algorithms",
    "cqlib_algorithms.operators",
):
    namespace = {}
    exec(f"from {module_name} import *", namespace)
    namespaces.append(sorted(name for name in namespace if name != "__builtins__"))

print(
    json.dumps(
        {
            "all": [
                namespaces[0] == sorted(cqlib_algorithms.__all__),
                namespaces[1] == sorted(algorithms.__all__),
                namespaces[2] == sorted(operators.__all__),
            ],
            "arithmetic_module": isinstance(arithmetic, ModuleType),
            "identities": [
                VQE is algorithms.VQE,
                arithmetic is operators.arithmetic,
                mutual_information is operators.mutual_information,
            ],
        }
    )
)
"""
    )

    assert result == {
        "all": [True, True, True],
        "arithmetic_module": True,
        "identities": [True, True, True],
    }


def test_lazy_exports_match_their_defining_objects_and_reject_unknown_names() -> None:
    result = _run_clean_process(
        """
import importlib
import json

import cqlib_algorithms
import cqlib_algorithms.algorithms as algorithms
import cqlib_algorithms.operators as operators

variational = importlib.import_module("cqlib_algorithms.algorithms.variational")
kernel = importlib.import_module("cqlib_algorithms.operators.quantum_kernel")
kernel_loss = importlib.import_module(
    "cqlib_algorithms.operators.quantum_kernel.balanced_kernel_alignment_loss"
)
trainable_kernel = importlib.import_module(
    "cqlib_algorithms.operators.quantum_kernel.trainable_fidelity_quantum_kernel"
)
information = importlib.import_module("cqlib_algorithms.operators.information")

unknown = []
for facade in (cqlib_algorithms, algorithms, operators):
    try:
        getattr(facade, "not_a_public_export")
    except AttributeError:
        unknown.append(True)
    else:
        unknown.append(False)

print(
    json.dumps(
        {
            "identities": [
                cqlib_algorithms.VQE is algorithms.VQE is variational.VQE,
                operators.FidelityQuantumKernel is kernel.FidelityQuantumKernel,
                operators.BalancedKernelAlignmentLoss
                is kernel_loss.BalancedKernelAlignmentLoss,
                operators.TrainableFidelityQuantumKernel
                is trainable_kernel.TrainableFidelityQuantumKernel,
                operators.mutual_information is information.mutual_information,
            ],
            "unknown": unknown,
            "version_cached": "__version__" in cqlib_algorithms.__dict__,
        }
    )
)
"""
    )

    assert result == {
        "identities": [True, True, True, True, True],
        "unknown": [True, True, True],
        "version_cached": True,
    }


def test_specialized_kernel_exports_target_their_defining_modules() -> None:
    result = _run_clean_process(
        """
import json

import cqlib_algorithms.operators as operators

print(
    json.dumps(
        {
            "balanced": operators._EXPORTS["BalancedKernelAlignmentLoss"][0],
            "trainable": operators._EXPORTS["TrainableFidelityQuantumKernel"][0],
            "quantum_kernel_loaded": (
                "cqlib_algorithms.operators.quantum_kernel" in __import__("sys").modules
            ),
        }
    )
)
"""
    )

    assert result == {
        "balanced": ".quantum_kernel.balanced_kernel_alignment_loss",
        "trainable": ".quantum_kernel.trainable_fidelity_quantum_kernel",
        "quantum_kernel_loaded": False,
    }
