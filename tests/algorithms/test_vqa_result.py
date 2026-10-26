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

"""Shared result validation and representation tests."""

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from cqlib import Circuit, Parameter
from cqlib_algorithms import QAOAResult, VQAResult, VQEResult
from cqlib_algorithms.protocols import OptimizerResult, OptimizerStatus


def _result(**changes: object) -> VQAResult:
    values: dict[str, object] = {
        "optimal_value": -0.5,
        "optimal_parameters": {"angle": 0.25},
        "optimal_circuit": Circuit(1),
        "optimizer_result": OptimizerResult(
            ("angle",), (0.25,), -0.5, 2, OptimizerStatus.MAX_ITERATIONS
        ),
    }
    values.update(changes)
    return VQAResult(**values)  # type: ignore[arg-type]


def test_result_aliases_and_default_minimum_semantics() -> None:
    result = _result()
    assert VQAResult is VQEResult is QAOAResult
    assert result.eigenvalue == result.optimal_value == -0.5
    assert result.objective_sense == "minimize"
    assert not hasattr(result, "vqe_result")


def test_parameter_mapping_is_copied_immutable_and_fields_are_frozen() -> None:
    source = {"angle": 0.25}
    result = _result(optimal_parameters=source)
    source["angle"] = 9.0
    assert result.optimal_parameters["angle"] == 0.25
    with pytest.raises(TypeError):
        result.optimal_parameters["angle"] = 1.0
    with pytest.raises(FrozenInstanceError):
        result.optimal_value = 0.0
    with pytest.raises(TypeError):
        hash(result)


def test_replace_runs_validation_and_preserves_standard_dataclass_behavior() -> None:
    result = _result()
    copied = replace(result)
    assert copied is not result
    assert copied.optimal_parameters == result.optimal_parameters
    maximum = replace(result, objective_sense="maximize", optimal_value=0.5)
    assert maximum.optimal_value == -maximum.optimizer_result.optimal_value
    with pytest.raises(ValueError, match="signed optimizer"):
        replace(result, optimal_value=4.0)
    with pytest.raises(ValueError, match="signed optimizer"):
        replace(result, objective_sense="maximize")
    with pytest.raises(ValueError, match="objective_sense"):
        replace(result, objective_sense="other")


def test_repr_is_compact_and_does_not_print_circuit_or_parameter_values() -> None:
    result = _result()
    assert repr(result) == (
        "VQAResult(optimal_value=-0.5, num_parameters=1, "
        "objective_sense='minimize', iterations=2, status='MAX_ITERATIONS')"
    )
    assert str(result) == repr(result)


@pytest.mark.parametrize("value", [True, "1", 1j])
def test_value_requires_real_non_boolean_number(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        _result(optimal_value=value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_value_requires_finiteness(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _result(optimal_value=value)


@pytest.mark.parametrize(
    "parameters, error",
    [
        ([], TypeError),
        ({1: 0.25}, TypeError),
        ({"angle": True}, TypeError),
        ({"angle": math.inf}, ValueError),
        ({"other": 0.25}, ValueError),
        ({"angle": 0.5}, ValueError),
    ],
)
def test_parameter_validation(parameters: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        _result(optimal_parameters=parameters)


def test_metadata_and_circuit_require_correct_types_and_bound_parameters() -> None:
    with pytest.raises(TypeError, match="OptimizerResult"):
        _result(optimizer_result=object())
    with pytest.raises(TypeError, match="cqlib.Circuit"):
        _result(optimal_circuit=object())
    circuit = Circuit(1)
    circuit.ry(0, Parameter("angle"))
    with pytest.raises(ValueError, match="fully bound"):
        _result(optimal_circuit=circuit)
