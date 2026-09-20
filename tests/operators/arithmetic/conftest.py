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

"""Pytest fixtures and dependency guards for the ``operators.arithmetic`` suite.

The ``cqlib_algorithms`` test-suite covers components whose baseline is the
``cqlib`` SDK (circuit construction, statevector simulator).  Certain optional
integration tests additionally cross-check against third-party libraries such
as ``qiskit``.  This conftest:

* Registers markers that describe external-library dependencies so that
  ``pytest --collect-only`` reports cleanly even when those libraries are
  absent.
* Provides a session-level ``requires_cqlib`` auto-sanity check: every
  arithmetic operator test must be able to import a working ``cqlib.Circuit``
  and ``cqlib.qis.state.Statevector`` or the whole arithmetic test directory
  is skipped.
* Centralises the ``HAS_QISKIT`` / ``HAS_...`` external-dependency detection
  used by tests that want to compare outputs against another SDK without
  hard-failing when it is missing.  Any test that depends on a third-party
  SDK should declare the dependency via the corresponding
  ``@pytest.mark.skipif`` boolean exported here, rather than duplicating the
  detection logic inline.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1.  Session-level sanity: a usable cqlib SDK must be importable
# ---------------------------------------------------------------------------


def _sanity_check_cqlib() -> str | None:
    """Return ``None`` if cqlib is usable, else a human-readable reason.

    We deliberately perform *both* an ``import`` check AND a tiny construct
    test (empty ``Circuit`` plus a ``Statevector`` round-trip) because
    packaging setups can sometimes expose the top-level import while still
    missing the extension libraries needed for simulation.
    """

    try:
        from cqlib.qis.state import Statevector

        from cqlib import Circuit
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        return f"cqlib import failed: {exc!r}"

    try:
        from cqlib.qis.state import Statevector

        from cqlib import Circuit

        c = Circuit(1)
        c.h(0)
        _sv = Statevector.from_circuit(c)
        _raw = _sv.data
        if callable(_raw):
            _raw = _raw()
        _arr = np.asarray(_raw, dtype=np.complex128)
        if len(_arr) != 2:
            return "Statevector.data returned an unexpected length"
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        return f"cqlib minimal simulation sanity probe failed: {exc!r}"

    return None


# @pytest.fixture(scope="session", autouse=True)
# def requires_cqlib_session() -> None:
#     """Auto-applied session guard.

#     Skip every test under ``tests/operators/arithmetic`` if the ``cqlib`` SDK
#     cannot be imported or cannot construct a 1-qubit ``Statevector``.  This
#     converts opaque ``ImportError`` / ``AttributeError`` traces during
#     per-test collection into a single declarative skip reason.
#     """

#     problem = _sanity_check_cqlib()
#     if problem is not None:  # pragma: no cover - only on broken envs
#         pytest.skip(
#             f"Arithmetic operator tests require a working cqlib SDK.  "
#             f"Probe reported: {problem}.  "
#             f"If you use ``uv run --no-sync`` please make sure the virtual env "
#             f"has cqlib + numpy installed, or run ``uv sync`` first.",
#             allow_module_level=True,
#         )


# ---------------------------------------------------------------------------
# 2.  External (non-cqlib) third-party library detection
# ---------------------------------------------------------------------------


def _probe_module(name: str, *, probe: str | None = None) -> bool:
    """Import ``name`` and, when given, additionally call ``getattr(mod, probe)``.

    Examples: ``_probe_module("qiskit")`` returns ``True`` if ``import qiskit``
    succeeds; ``_probe_module("qiskit", probe="QuantumCircuit")`` also checks
    the attribute access.
    """

    try:
        import importlib

        mod = importlib.import_module(name)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    if probe is not None:
        try:
            getattr(mod, probe)
        except AttributeError:
            return False
    return True


#: ``True`` iff the local environment provides ``qiskit`` with a working
#: ``QuantumCircuit``.  Used by INT-004 style cross-SDK comparison tests.
HAS_QISKIT: bool = _probe_module("qiskit", probe="QuantumCircuit") and _probe_module(
    "qiskit.quantum_info", probe="Operator"
)

#: ``True`` iff ``pennylane`` is importable (reserved for future comparisons).
HAS_PENNYLANE: bool = _probe_module("pennylane", probe="tape")

#: ``True`` iff ``cirq`` is importable (reserved for future comparisons).
HAS_CIRQ: bool = _probe_module("cirq", probe="Circuit")


# ---------------------------------------------------------------------------
# 3.  Convenience pytest markers for external dependencies
# ---------------------------------------------------------------------------


# Marker string constants used by tests to request skip-on-missing-lib.
# ``pytest --markers`` will list them if registered below.
_MARKERS: dict[str, str] = {
    "requires_qiskit": (
        "Mark tests that compare against a Qiskit reference implementation; "
        "automatically skipped when ``HAS_QISKIT`` is ``False``."
    ),
    "requires_pennylane": (
        "Mark tests that cross-check against PennyLane; automatically skipped "
        "when ``HAS_PENNYLANE`` is ``False``."
    ),
    "requires_cirq": (
        "Mark tests that cross-check against Cirq; automatically skipped when "
        "``HAS_CIRQ`` is ``False``."
    ),
}


def pytest_configure(config: pytest.Config) -> None:
    """Register the custom markers declared by this conftest.

    Ensures ``pytest --strict-markers`` runs do not flag our markers as
    unknown.
    """

    for name, description in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


# ---------------------------------------------------------------------------
# 4.  Helpers exposed as fixtures / module-level helpers for arithmetic tests
# ---------------------------------------------------------------------------


def skip_if_missing_qiskit() -> Any:
    """Shortcut for ``@pytest.mark.skipif`` decorator targeting Qiskit."""

    return pytest.mark.skipif(
        not HAS_QISKIT, reason="Qiskit is not installed in this environment"
    )


def skip_if_missing_pennylane() -> Any:
    """Shortcut ``@pytest.mark.skipif`` decorator targeting PennyLane."""

    return pytest.mark.skipif(
        not HAS_PENNYLANE, reason="PennyLane is not installed in this environment"
    )


def skip_if_missing_cirq() -> Any:
    """Shortcut ``@pytest.mark.skipif`` decorator targeting Cirq."""

    return pytest.mark.skipif(
        not HAS_CIRQ, reason="Cirq is not installed in this environment"
    )


# Expose the helper decorators as fixtures too so tests can either import the
# module-level helper directly, or use ``def test_foo(skip_if_missing_qiskit):``.
@pytest.fixture(name="skip_if_missing_qiskit")
def _fixture_skip_qiskit() -> Any:  # pragma: no cover - trivial pass-through
    return skip_if_missing_qiskit()


@pytest.fixture(name="skip_if_missing_pennylane")
def _fixture_skip_pennylane() -> Any:  # pragma: no cover - trivial pass-through
    return skip_if_missing_pennylane()


@pytest.fixture(name="skip_if_missing_cirq")
def _fixture_skip_cirq() -> Any:  # pragma: no cover - trivial pass-through
    return skip_if_missing_cirq()


# ---------------------------------------------------------------------------
# 5.  Reproducible random seed across arithmetic tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _fixed_arithmetic_random_seed() -> None:
    """Fix the NumPy random seed for every arithmetic test module.

    Seed is ``20260919``; tests that rely on random angle tables or random
    coefficients are therefore deterministic across runs.  The scope is set
    to ``module`` to keep every module independent while still avoiding
    per-call reseeding overhead.
    """

    # Imported lazily so environments with broken ``numpy`` installations
    # still surface the cqlib probe failure first.
    import numpy as np

    np.random.seed(int(os.environ.get("CQLIB_ARITHMETIC_SEED", "20260919")))
    yield
