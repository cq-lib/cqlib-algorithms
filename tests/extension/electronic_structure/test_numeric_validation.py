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

"""Behavior and placement tests for electronic-structure numeric helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cqlib_algorithms.extension.electronic_structure._numeric import (
    immutable_complex_array,
    immutable_real_array,
    validate_finite_real,
    validate_strict_int,
)

_CHEMISTRY_SOURCE = (
    Path(__file__).parents[3]
    / "src"
    / "cqlib_algorithms"
    / "extension"
    / "electronic_structure"
)


@pytest.mark.parametrize(
    ("factory", "dtype"),
    [
        (immutable_real_array, np.dtype(np.float64)),
        (immutable_complex_array, np.dtype(np.complex128)),
    ],
)
def test_immutable_arrays_are_typed_readonly_defensive_snapshots(factory, dtype):
    source = np.array([[1, 2], [3, 4]])
    result = factory(source, name="value")
    source[0, 0] = 99

    assert result.dtype == dtype
    assert result.shape == (2, 2)
    assert not result.flags.writeable
    assert result[0, 0] == 1
    with pytest.raises(ValueError, match="read-only"):
        result[0, 0] = 5


@pytest.mark.parametrize("value", [[[True]], [["1"]], [[1], [2, 3]]])
@pytest.mark.parametrize("factory", [immutable_real_array, immutable_complex_array])
def test_immutable_arrays_reject_boolean_string_and_ragged_inputs(factory, value):
    with pytest.raises(TypeError):
        factory(value, name="value")


@pytest.mark.parametrize("value", [[[np.nan]], [[np.inf]], [[-np.inf]]])
@pytest.mark.parametrize("factory", [immutable_real_array, immutable_complex_array])
def test_immutable_arrays_reject_nonfinite_inputs(factory, value):
    with pytest.raises(ValueError, match="finite"):
        factory(value, name="value")


def test_real_array_rejects_complex_values() -> None:
    with pytest.raises(TypeError, match="real values"):
        immutable_real_array([1.0 + 0.0j], name="value")


@pytest.mark.parametrize("value", [True, np.int64(1), "1", 1.0])
def test_strict_integer_validation_rejects_non_builtin_integers(value) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        validate_strict_int(value, name="value")


def test_strict_integer_validation_honors_minimum_and_optional_bound() -> None:
    validate_strict_int(0, name="value")
    validate_strict_int(-3, name="value", minimum=None)
    with pytest.raises(ValueError, match="at least 2"):
        validate_strict_int(1, name="value", minimum=2)


@pytest.mark.parametrize("value", [0, 1.5, np.int64(2), np.float64(-3.5)])
def test_finite_real_validation_accepts_supported_scalars(value) -> None:
    validate_finite_real(value, name="value")


@pytest.mark.parametrize("value", [True, "1", 1.0j])
def test_finite_real_validation_rejects_non_real_scalars(value) -> None:
    with pytest.raises(TypeError, match="real number"):
        validate_finite_real(value, name="value")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_finite_real_validation_rejects_nonfinite_scalars(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_finite_real(value, name="value")


def test_legacy_validation_package_and_imports_are_absent() -> None:
    assert not (_CHEMISTRY_SOURCE / "validation").exists()
    for source in _CHEMISTRY_SOURCE.rglob("*.py"):
        assert "validation.numerical" not in source.read_text(encoding="utf-8")
