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

"""Tests for the optimizer protocol and immutable value types."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.protocols import (
    EstimatorRequest,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
)


def _estimator_request():
    first = Parameter("first")
    fixed = Parameter("fixed")
    circuit = Circuit(1)
    circuit.ry(0, first + fixed)
    return EstimatorRequest(
        circuit,
        PauliString.from_str("Z"),
        {"first": 0.2, "fixed": -0.1},
    )


def test_optimizer_is_abstract() -> None:
    with pytest.raises(TypeError):
        Optimizer()  # type: ignore[abstract]


def test_request_accepts_an_ordered_trainable_subset() -> None:
    request = OptimizerRequest(_estimator_request(), ("first",))
    assert request.parameter_names == ("first",)
    assert request.estimator_request.parameter_bindings["fixed"] == -0.1


def test_empty_parameter_subset_is_valid() -> None:
    assert OptimizerRequest(_estimator_request(), ()).parameter_names == ()


@pytest.mark.parametrize(
    ("names", "error", "message"),
    [
        (["first"], TypeError, "tuple"),
        ((1,), TypeError, "strings"),
        (("first", "first"), ValueError, "unique"),
        (("unknown",), ValueError, "bound"),
    ],
)
def test_request_rejects_invalid_names(names, error, message) -> None:
    with pytest.raises(error, match=message):
        OptimizerRequest(_estimator_request(), names)


def test_request_rejects_wrong_nested_type() -> None:
    with pytest.raises(TypeError, match="EstimatorRequest"):
        OptimizerRequest(object(), ())  # type: ignore[arg-type]


def test_request_revalidates_nested_mutation() -> None:
    request = OptimizerRequest(_estimator_request(), ("first",))
    extra = Parameter("extra")
    request.estimator_request.circuit.rz(0, extra)
    with pytest.raises(ValueError, match="missing"):
        request.validate()


def test_result_normalizes_real_values_and_is_frozen() -> None:
    result = OptimizerResult(
        ("theta",),
        (np.float64(0.25),),
        np.float64(-0.5),
        3,
        OptimizerStatus.MAX_ITERATIONS,
    )
    assert result.optimal_parameters == (0.25,)
    assert result.optimal_value == -0.5
    assert not result.converged
    with pytest.raises(FrozenInstanceError):
        result.iterations = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("status", "converged"),
    [
        (OptimizerStatus.CONVERGED, True),
        (OptimizerStatus.STEP_TOLERANCE, True),
        (OptimizerStatus.NO_PARAMETERS, True),
        (OptimizerStatus.MAX_EVALUATIONS, False),
        (OptimizerStatus.MAX_ITERATIONS, False),
    ],
)
def test_converged_property(status, converged) -> None:
    result = OptimizerResult((), (), 1.0, 0, status)
    assert result.converged is converged


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((["x"], (0.0,), 0.0, 0, OptimizerStatus.NO_PARAMETERS), TypeError, "tuple"),
        (((1,), (0.0,), 0.0, 0, OptimizerStatus.NO_PARAMETERS), TypeError, "strings"),
        (
            (("x", "x"), (0.0, 0.0), 0.0, 0, OptimizerStatus.NO_PARAMETERS),
            ValueError,
            "unique",
        ),
        ((("x",), (), 0.0, 0, OptimizerStatus.NO_PARAMETERS), ValueError, "equal"),
        (
            (("x",), (True,), 0.0, 0, OptimizerStatus.NO_PARAMETERS),
            TypeError,
            "real",
        ),
        (
            (("x",), (float("inf"),), 0.0, 0, OptimizerStatus.NO_PARAMETERS),
            ValueError,
            "finite",
        ),
        (((), (), True, 0, OptimizerStatus.NO_PARAMETERS), TypeError, "real"),
        (
            ((), (), float("nan"), 0, OptimizerStatus.NO_PARAMETERS),
            ValueError,
            "finite",
        ),
        (((), (), 0.0, True, OptimizerStatus.NO_PARAMETERS), TypeError, "integer"),
        (((), (), 0.0, -1, OptimizerStatus.NO_PARAMETERS), ValueError, "non-negative"),
        (((), (), 0.0, 0, "done"), TypeError, "OptimizerStatus"),
    ],
)
def test_result_rejects_invalid_fields(args, error, message) -> None:
    with pytest.raises(error, match=message):
        OptimizerResult(*args)
