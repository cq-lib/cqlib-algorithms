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

"""Smoke checks for the installed distribution."""

import inspect
from importlib import import_module
from importlib.metadata import metadata
from importlib.resources import files

import pytest
from cqlib.error_mitigation import Estimator as ErrorMitigationEstimator

import cqlib_algorithms
from cqlib_algorithms import (
    QAOA,
    QFT,
    VQE,
    BernsteinVazirani,
    BernsteinVaziraniRequest,
    BernsteinVaziraniResult,
    DeutschJozsa,
    DeutschJozsaRequest,
    DeutschJozsaResult,
    QAOAResult,
    VQAResult,
    VQEResult,
)
from cqlib_algorithms.algorithms import QAOA as AlgorithmQAOA
from cqlib_algorithms.algorithms import VQE as AlgorithmVQE
from cqlib_algorithms.algorithms import (
    BernsteinVazirani as AlgorithmBernsteinVazirani,
)
from cqlib_algorithms.algorithms import (
    BernsteinVaziraniRequest as AlgorithmBernsteinVaziraniRequest,
)
from cqlib_algorithms.algorithms import (
    BernsteinVaziraniResult as AlgorithmBernsteinVaziraniResult,
)
from cqlib_algorithms.algorithms import (
    DeutschJozsa as AlgorithmDeutschJozsa,
)
from cqlib_algorithms.algorithms import (
    DeutschJozsaRequest as AlgorithmDeutschJozsaRequest,
)
from cqlib_algorithms.algorithms import (
    DeutschJozsaResult as AlgorithmDeutschJozsaResult,
)
from cqlib_algorithms.algorithms import QAOAResult as AlgorithmQAOAResult
from cqlib_algorithms.algorithms import VQEResult as AlgorithmVQEResult
from cqlib_algorithms.protocols import (
    Ansatz,
    Estimator,
    EstimatorRequest,
    EstimatorResult,
    Gradient,
    GradientRequest,
    GradientResult,
    Optimizer,
    OptimizerRequest,
    OptimizerResult,
    OptimizerStatus,
    Sampler,
    SamplerRequest,
    StatePreparation,
)
from cqlib_algorithms.transforms import QFT as TransformQFT

PUBLIC_COMPONENTS = {
    "Ansatz": Ansatz,
    "BernsteinVazirani": BernsteinVazirani,
    "BernsteinVaziraniRequest": BernsteinVaziraniRequest,
    "BernsteinVaziraniResult": BernsteinVaziraniResult,
    "DeutschJozsa": DeutschJozsa,
    "DeutschJozsaRequest": DeutschJozsaRequest,
    "DeutschJozsaResult": DeutschJozsaResult,
    "Estimator": Estimator,
    "EstimatorRequest": EstimatorRequest,
    "EstimatorResult": EstimatorResult,
    "Gradient": Gradient,
    "GradientRequest": GradientRequest,
    "GradientResult": GradientResult,
    "Optimizer": Optimizer,
    "OptimizerRequest": OptimizerRequest,
    "OptimizerResult": OptimizerResult,
    "OptimizerStatus": OptimizerStatus,
    "QAOA": QAOA,
    "QAOAResult": QAOAResult,
    "Sampler": Sampler,
    "SamplerRequest": SamplerRequest,
    "StatePreparation": StatePreparation,
    "VQE": VQE,
    "VQEResult": VQEResult,
    "VQAResult": VQAResult,
}


def test_distribution_metadata():
    distribution = metadata("cqlib-algorithms")
    assert distribution["Requires-Python"] == ">=3.10"
    assert cqlib_algorithms.__version__ == distribution["Version"]


def test_algorithms_package_is_importable():
    assert import_module("cqlib_algorithms.algorithms") is not None


def test_protocols_package_is_importable():
    assert import_module("cqlib_algorithms.protocols") is not None


def test_transforms_package_is_importable():
    assert import_module("cqlib_algorithms.transforms") is not None


def test_typing_marker_is_packaged():
    assert files("cqlib_algorithms").joinpath("py.typed").is_file()


@pytest.mark.parametrize(("name", "component"), PUBLIC_COMPONENTS.items())
def test_stable_components_are_exported_from_package(name, component):
    assert getattr(cqlib_algorithms, name) is component


def test_package_all_matches_the_stable_public_api():
    assert set(cqlib_algorithms.__all__) == {"__version__", "QFT", *PUBLIC_COMPONENTS}


def test_estimator_contract_is_distinct_from_the_core_callback_alias():
    assert cqlib_algorithms.Estimator is not ErrorMitigationEstimator


def test_component_contracts_are_abstract_base_classes():
    for component in (
        Ansatz,
        Estimator,
        Gradient,
        Optimizer,
        Sampler,
        StatePreparation,
    ):
        assert inspect.isabstract(component)


def test_qft_is_exported_from_public_namespaces():
    assert QFT is TransformQFT


def test_vqe_is_exported_from_public_namespaces():
    assert VQE is AlgorithmVQE
    assert VQEResult is AlgorithmVQEResult


def test_qaoa_is_exported_from_public_namespaces():
    assert QAOA is AlgorithmQAOA
    assert QAOAResult is AlgorithmQAOAResult


def test_deutsch_jozsa_is_exported_from_public_namespaces():
    assert DeutschJozsa is AlgorithmDeutschJozsa
    assert DeutschJozsaRequest is AlgorithmDeutschJozsaRequest
    assert DeutschJozsaResult is AlgorithmDeutschJozsaResult


def test_bernstein_vazirani_is_exported_from_public_namespaces():
    assert BernsteinVazirani is AlgorithmBernsteinVazirani
    assert BernsteinVaziraniRequest is AlgorithmBernsteinVaziraniRequest
    assert BernsteinVaziraniResult is AlgorithmBernsteinVaziraniResult
