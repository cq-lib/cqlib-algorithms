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

"""Tests for the phase-estimation HHL linear-system solver."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
from cqlib import Circuit
from cqlib.device import ExecutionResult
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms.algorithms import (
    HHL,
    HHLComponents,
    HHLRequest,
    HHLResult,
    LinearSystemProblem,
)
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    PhaseEstimationCircuit,
    StatevectorSampler,
)
from cqlib_algorithms.operators.arithmetic import ExactReciprocal
from cqlib_algorithms.protocols import (
    Sampler,
    SamplerRequest,
    validate_sampler_result,
)

_EVALUATION_QUBITS = 4
_SCALING_TOLERANCE = 1e-9

_RECORDS: list[str] = []


class _RecordingEncoding(AmplitudeEncoding):
    """State preparation recording that the assembly instantiated it."""

    def __init__(self, features):
        super().__init__(features)
        _RECORDS.append("state_preparation")


class _RecordingPhaseEstimation(PhaseEstimationCircuit):
    """Phase estimation recording its forward and adjoint instantiations."""

    def __init__(
        self,
        unitary,
        num_evaluation_qubits,
        *,
        approximation_degree=0,
        inverse=False,
    ):
        super().__init__(
            unitary,
            num_evaluation_qubits,
            approximation_degree=approximation_degree,
            inverse=inverse,
        )
        _RECORDS.append(f"phase_estimation:{inverse}")


class _RecordingInversion(ExactReciprocal):
    """Reciprocal rotation recording that the assembly instantiated it."""

    def __init__(self, num_state_qubits, scaling, **kwargs):
        super().__init__(num_state_qubits, scaling, **kwargs)
        _RECORDS.append("eigenvalue_inversion")


def _diag_problem(eigenvalues, rhs):
    operator = np.diag(np.asarray(eigenvalues, dtype=np.complex128))
    return LinearSystemProblem(operator=operator, rhs=np.asarray(rhs, dtype=np.complex128))


def _request(
    problem,
    *,
    num_evaluation_qubits=_EVALUATION_QUBITS,
    scaling=None,
    shots=4096,
    **kwargs,
):
    if scaling is None:
        scaling = 2.0**-num_evaluation_qubits
    return HHLRequest(
        problem,
        num_evaluation_qubits,
        scaling,
        shots,
        **kwargs,
    )


def _analytic_solution(operator, rhs):
    """Return the normalized ``|A**-1 b>`` basis distribution."""
    matrix = np.asarray(operator, dtype=np.complex128)
    vector = np.asarray(rhs, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    solution = np.linalg.solve(matrix, vector)
    solution = solution / np.linalg.norm(solution)
    return np.abs(solution) ** 2


def _analytic_success(operator, rhs, alpha, scaling):
    """Return the exact post-selection probability of the ideal circuit."""
    matrix = np.asarray(operator, dtype=np.complex128)
    values, vectors = np.linalg.eigh(matrix)
    vector = np.asarray(rhs, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    weights = np.abs(vectors.conj().T @ vector) ** 2
    return float(np.sum(weights * (scaling * alpha / values) ** 2))


def _joint_distribution(circuit, num_evaluation_qubits, system_width):
    """Return the exact joint distribution of (system index, ancilla bit)."""
    state = np.asarray(circuit.to_matrix(), dtype=np.complex128)[:, 0]
    weights = np.abs(state) ** 2
    indices = np.arange(weights.size)
    system_bits = (indices >> num_evaluation_qubits) & ((1 << system_width) - 1)
    ancilla_bits = (indices >> (num_evaluation_qubits + system_width)) & 1
    joint: dict[tuple[int, int], float] = {}
    for system, ancilla, weight in zip(system_bits, ancilla_bits, weights):
        key = (int(system), int(ancilla))
        joint[key] = joint.get(key, 0.0) + float(weight)
    return joint


def _analytic_joint(operator, rhs, alpha, scaling, num_evaluation_qubits):
    """Return the grid-exact joint distribution of an ideal HHL circuit.

    The entries are exact only when every ``lambda_j / alpha`` is an integer
    multiple of ``2**-m``, because the phase-estimation QFT is then free of
    discretization error.
    """
    matrix = np.asarray(operator, dtype=np.complex128)
    values, vectors = np.linalg.eigh(matrix)
    vector = np.asarray(rhs, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    weights = np.abs(vectors.conj().T @ vector) ** 2
    joint: dict[tuple[int, int], float] = {}
    for eigenvalue, eigenvector, weight in zip(values, vectors.T, weights):
        grid_point = round(float(eigenvalue) / alpha * (1 << num_evaluation_qubits))
        amplitude = scaling * (1 << num_evaluation_qubits) / grid_point
        for index, component in enumerate(eigenvector):
            probability = weight * abs(component) ** 2
            joint[(index, 0)] = joint.get((index, 0), 0.0) + probability * (
                1.0 - amplitude**2
            )
            joint[(index, 1)] = joint.get((index, 1), 0.0) + probability * amplitude**2
    return joint


def _success_probability(joint):
    return sum(weight for (_, ancilla), weight in joint.items() if ancilla == 1)


def _dense_distribution(joint, system_width):
    """Return the post-selected system distribution as a dense vector."""
    success = _success_probability(joint)
    vector = np.zeros(1 << system_width, dtype=np.float64)
    if success == 0.0:
        return vector
    for (index, ancilla), weight in joint.items():
        if ancilla == 1:
            vector[index] += weight / success
    return vector


def _exact_distribution(circuit, num_evaluation_qubits, system_width):
    joint = _joint_distribution(circuit, num_evaluation_qubits, system_width)
    return _dense_distribution(joint, system_width)


def _assert_solution_matches_analytic(result, operator, rhs, shots, sigmas=3.0):
    """Assert a sampled solution matches the analytic distribution.

    The post-selected ensemble holds ``success_probability * shots`` samples,
    so the standard error of each observed probability ``p`` is
    ``sqrt(p * (1 - p) / n)``. The tolerance is ``sigmas`` standard errors,
    which keeps finite-shot comparisons from being seed-dependent.
    """
    expected = _analytic_solution(operator, rhs)
    observed = dict(result.solution_probabilities)
    post_selected = max(result.success_probability * shots, 1.0)
    for index, probability in enumerate(expected):
        standard_error = math.sqrt(
            probability * (1.0 - probability) / post_selected
        )
        assert observed.get(index, 0.0) == pytest.approx(
            probability, abs=sigmas * standard_error
        )


def _counts_key(system_index, ancilla, system_width):
    """Return the little-endian count key of one measured outcome."""
    low_mask = (1 << system_width) - 1
    outcome = (system_index & low_mask) | ((ancilla & 1) << system_width)
    return format(outcome, f"0{system_width + 1}b")


class _ScriptedSampler(Sampler):
    """Sampler returning caller-controlled counts for contract tests."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.calls = 0

    def run(self, requests):
        self.calls += 1
        return [
            ExecutionResult.from_counts(
                "scripted",
                list(request.qubits),
                request.shots,
                len(request.qubits),
                counts,
                "fake-backend",
            )
            for request, counts in zip(requests, self._scripts)
        ]


class _InvalidResultSampler(Sampler):
    """Return one result with a selected Sampler-contract violation."""

    def __init__(self, mode):
        self._mode = mode

    def run(self, requests):
        request = requests[0]
        width = len(request.qubits)
        if self._mode == "queued":
            return [
                ExecutionResult(
                    "queued",
                    list(request.qubits),
                    request.shots,
                    width,
                )
            ]
        if self._mode == "failed":
            result = ExecutionResult(
                "failed",
                list(request.qubits),
                request.shots,
                width,
            )
            result.fail("deliberate failure", 1)
            return [result]
        if self._mode == "width":
            return [
                ExecutionResult.from_counts(
                    "width", [request.qubits[0]], request.shots, 1, {"0": 4}
                )
            ]
        if self._mode == "qubits":
            return [
                ExecutionResult.from_counts(
                    "qubits",
                    list(reversed(request.qubits)),
                    request.shots,
                    width,
                    {"0" * width: 4},
                )
            ]
        raise AssertionError("unsupported invalid-result mode")


# ---------------------------------------------------------------------------
# LinearSystemProblem
# ---------------------------------------------------------------------------


def test_problem_snapshots_operator_and_normalizes_dtype():
    operator = np.diag([1.0, 0.5])
    problem = LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])

    operator[0, 0] = 99.0

    assert problem.operator.dtype == np.complex128
    assert problem.operator[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("shape", "num_qubits"),
    [((2, 2), 1), ((4, 4), 2), ((8, 8), 3)],
)
def test_problem_num_qubits_matches_operator_dimension(shape, num_qubits):
    operator = np.eye(shape[0], dtype=np.complex128)
    problem = LinearSystemProblem(
        operator=operator,
        rhs=np.ones(shape[0], dtype=np.complex128),
    )

    assert problem.num_qubits == num_qubits


def test_problem_copies_rhs_vector_verbatim():
    problem = _diag_problem([1.0, 0.5], [3.0, 4.0])

    # The container snapshots the vector as given; AmplitudeEncoding performs
    # the unit normalization when the state preparation circuit is built.
    np.testing.assert_allclose(np.asarray(problem.rhs), [3.0, 4.0])


def test_problem_rejects_zero_rhs():
    with pytest.raises(ValueError, match="rhs must not be the zero vector"):
        _diag_problem([1.0, 0.5], [0.0, 0.0])


def test_problem_rejects_rhs_length_mismatch():
    with pytest.raises(ValueError, match="rhs must have length"):
        _diag_problem([1.0, 0.5], [1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("rhs", ["rhs", b"rhs", 1])
def test_problem_rejects_non_sequence_rhs(rhs):
    with pytest.raises(TypeError, match="rhs must be a sequence"):
        LinearSystemProblem(operator=np.diag([1.0, 0.5]), rhs=rhs)


def test_problem_rejects_single_entry_rhs():
    operator = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="rhs must contain at least two values"):
        LinearSystemProblem(operator=operator, rhs=[1.0])


def test_problem_snapshots_rhs_circuit():
    preparation = Circuit(1)
    preparation.h(0)
    problem = LinearSystemProblem(operator=np.diag([1.0, 0.5]), rhs=preparation)

    preparation.x(0)

    assert problem.rhs.num_qubits == 1
    assert problem.rhs.to_matrix()[0, 0] == pytest.approx(1.0 / math.sqrt(2.0))


def test_problem_rejects_rhs_circuit_width():
    preparation = Circuit(2)
    preparation.h(0)
    with pytest.raises(ValueError, match="rhs circuit width must equal the system width"):
        LinearSystemProblem(operator=np.diag([1.0, 0.5]), rhs=preparation)


def test_problem_rejects_non_square_operator():
    with pytest.raises(ValueError, match="operator must be a square matrix"):
        LinearSystemProblem(operator=np.zeros((2, 3)), rhs=[1.0, 1.0])


def test_problem_rejects_non_power_of_two_operator():
    with pytest.raises(ValueError, match="must be a power of two"):
        LinearSystemProblem(operator=np.eye(3), rhs=[1.0, 1.0, 1.0])


def test_problem_rejects_non_finite_operator():
    operator = np.diag([1.0, np.nan])
    with pytest.raises(ValueError, match="operator must be finite"):
        LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])


def test_problem_rejects_singular_operator():
    operator = np.diag([1.0, 0.0])
    with pytest.raises(ValueError, match="operator must be non-singular"):
        LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])


@pytest.mark.parametrize("operator", [None, "A", 2])
def test_problem_rejects_non_operator_types(operator):
    with pytest.raises(
        TypeError, match="operator must be a numpy.ndarray or a Hamiltonian"
    ):
        LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])


def test_problem_accepts_hamiltonian_operator():
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("I"), 1.5)
    hamiltonian.add_term(PauliString.from_str("Z"), 0.5)
    problem = LinearSystemProblem(operator=hamiltonian, rhs=[1.0, 1.0])

    assert problem.num_qubits == 1
    assert isinstance(problem.operator, Hamiltonian)


def test_problem_container_accepts_structurally_valid_non_solver_operator():
    # The container is shared with non-HHL solvers, so it must not impose the
    # algorithm-specific positive-definiteness constraint.
    operator = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.complex128)
    problem = LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])

    assert problem.num_qubits == 1


def test_problem_validate_is_idempotent():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])

    problem.validate()
    problem.validate()


def test_problem_is_frozen():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])

    with pytest.raises(AttributeError):
        problem.rhs = [1.0, 0.0]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HHLComponents
# ---------------------------------------------------------------------------


def test_components_default_registry():
    components = HHLComponents()

    assert components.state_preparation is AmplitudeEncoding
    assert components.phase_estimation is PhaseEstimationCircuit
    assert components.eigenvalue_inversion is ExactReciprocal


def test_components_is_frozen_and_slotted():
    components = HHLComponents()

    assert not hasattr(components, "__dict__")
    with pytest.raises(AttributeError):
        components.state_preparation = AmplitudeEncoding  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    [AmplitudeEncoding([1.0, 0.0]), object, ExactReciprocal],
)
def test_components_reject_non_state_preparation_classes(value):
    with pytest.raises(
        TypeError, match="state_preparation must be a StatePreparation subclass"
    ):
        HHLComponents(state_preparation=value)


@pytest.mark.parametrize("value", [object, AmplitudeEncoding])
def test_components_reject_non_phase_estimation_classes(value):
    with pytest.raises(
        TypeError, match="phase_estimation must be a PhaseEstimationCircuit subclass"
    ):
        HHLComponents(phase_estimation=value)


@pytest.mark.parametrize("value", [object, PhaseEstimationCircuit])
def test_components_reject_non_inversion_classes(value):
    with pytest.raises(
        TypeError, match="eigenvalue_inversion must be an ExactReciprocal subclass"
    ):
        HHLComponents(eigenvalue_inversion=value)


def test_algorithm_rejects_non_registry_components():
    with pytest.raises(TypeError, match="components must be an HHLComponents instance"):
        HHL(StatevectorSampler(), components=object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HHLRequest
# ---------------------------------------------------------------------------


def test_request_defaults():
    request = _request(_diag_problem([1.0, 0.5], [1.0, 1.0]))

    assert request.approximation_degree == 0
    assert request.bound is None
    assert request.seed is None


def test_request_rejects_non_problem():
    with pytest.raises(TypeError, match="problem must be a LinearSystemProblem"):
        HHLRequest("problem", 4, 2.0**-4, 8)  # type: ignore[arg-type]


@pytest.mark.parametrize("width", [True, 1.5, "4"])
def test_request_rejects_non_integer_evaluation_width(width):
    # Pass an explicit scaling so the helper never evaluates 2.0**-width and
    # the request validation sees the malformed width directly.
    with pytest.raises(TypeError, match="num_evaluation_qubits must be an integer"):
        _request(
            _diag_problem([1.0, 0.5], [1.0, 1.0]),
            num_evaluation_qubits=width,
            scaling=2.0**-_EVALUATION_QUBITS,
        )


@pytest.mark.parametrize("width", [0, -1])
def test_request_rejects_non_positive_evaluation_width(width):
    with pytest.raises(ValueError, match="num_evaluation_qubits must be greater than zero"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), num_evaluation_qubits=width)


def test_request_rejects_excessive_evaluation_width():
    with pytest.raises(ValueError, match="must not exceed 52"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), num_evaluation_qubits=53)


@pytest.mark.parametrize("scaling", [0.0, -1.0, float("nan"), float("inf")])
def test_request_rejects_invalid_scaling(scaling):
    with pytest.raises(ValueError, match="scaling must be positive and finite"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), scaling=scaling)


def test_request_rejects_scaling_beyond_valid_bound():
    with pytest.raises(ValueError, match="scaling must not exceed"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), scaling=1.0)


@pytest.mark.parametrize("scaling", [True, "s", complex(1.0, 0.0)])
def test_request_rejects_non_real_scaling(scaling):
    with pytest.raises(TypeError, match="scaling must be a real number"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), scaling=scaling)


def test_request_requires_explicit_scaling():
    with pytest.raises(TypeError):
        HHLRequest(_diag_problem([1.0, 0.5], [1.0, 1.0]), 4, shots=8)


@pytest.mark.parametrize("shots", [True, 1.5, "8"])
def test_request_rejects_non_integer_shots(shots):
    with pytest.raises(TypeError, match="shots must be an integer"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=shots)


@pytest.mark.parametrize("shots", [0, -1])
def test_request_rejects_non_positive_shots(shots):
    with pytest.raises(ValueError, match="shots must be greater than zero"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=shots)


def test_request_rejects_shots_beyond_exact_integer_range():
    with pytest.raises(ValueError, match="shots must not exceed"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=1 << 53)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_request_rejects_non_integer_seed(seed):
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), seed=seed)


def test_request_rejects_negative_seed():
    with pytest.raises(ValueError, match="seed must be non-negative"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), seed=-1)


@pytest.mark.parametrize("degree", [True, 1.5, "1"])
def test_request_rejects_non_integer_approximation_degree(degree):
    with pytest.raises(TypeError, match="approximation_degree must be an integer"):
        _request(
            _diag_problem([1.0, 0.5], [1.0, 1.0]),
            approximation_degree=degree,
        )


@pytest.mark.parametrize("degree", [-1, _EVALUATION_QUBITS])
def test_request_rejects_out_of_range_approximation_degree(degree):
    with pytest.raises(ValueError, match="approximation_degree must satisfy"):
        _request(
            _diag_problem([1.0, 0.5], [1.0, 1.0]),
            approximation_degree=degree,
        )


def test_request_rejects_non_hermitian_operator():
    operator = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.complex128)
    problem = LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])

    with pytest.raises(ValueError, match="operator must be Hermitian"):
        _request(problem)


def test_request_rejects_non_positive_definite_operator():
    problem = _diag_problem([1.0, -0.5], [1.0, 1.0])

    with pytest.raises(ValueError, match="operator must be positive definite"):
        _request(problem)


def test_request_rejects_bound_below_largest_eigenvalue():
    with pytest.raises(
        ValueError, match="bound must not be smaller than the largest eigenvalue"
    ):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), bound=0.5)


def test_request_accepts_bound_equal_to_largest_eigenvalue():
    with pytest.warns(UserWarning, match="rounds onto the zero grid point"):
        request = _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), bound=1.0)

    assert request.bound == 1.0


@pytest.mark.parametrize("bound", [0.0, -1.0, float("nan"), float("inf")])
def test_request_rejects_invalid_bound(bound):
    with pytest.raises(ValueError, match="bound must be positive and finite"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), bound=bound)


@pytest.mark.parametrize("bound", [True, "2"])
def test_request_rejects_non_real_bound(bound):
    with pytest.raises(TypeError, match="bound must be a real number or None"):
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), bound=bound)


def test_request_is_frozen():
    request = _request(_diag_problem([1.0, 0.5], [1.0, 1.0]))

    with pytest.raises(AttributeError):
        request.bound = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HHLResult
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 2, 3, 6])
def test_result_num_unitary_applications(width):
    result = HHLResult(
        success_probability=0.5,
        solution_probabilities={0: 1.0},
        spectral_bound=2.0,
        scaling=2.0**-width,
        num_evaluation_qubits=width,
        num_system_qubits=1,
        shots=8,
    )

    assert result.num_unitary_applications == 2 * ((1 << width) - 1)


def test_result_num_unitary_applications_is_derived_not_stored():
    names = {field.name for field in dataclasses.fields(HHLResult)}

    assert "num_unitary_applications" not in names


def test_result_rejects_unnormalized_distribution():
    with pytest.raises(ValueError, match="solution probabilities must sum to one"):
        HHLResult(
            success_probability=0.5,
            solution_probabilities={0: 0.5},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=1,
            shots=8,
        )


@pytest.mark.parametrize("index", [-1, 4, 16])
def test_result_rejects_out_of_range_indices(index):
    with pytest.raises(ValueError, match="solution indices must be valid system indices"):
        HHLResult(
            success_probability=0.5,
            solution_probabilities={index: 1.0},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=2,
            shots=8,
        )


@pytest.mark.parametrize("probability", [0.0, -0.5, 1.5])
def test_result_rejects_out_of_range_probabilities(probability):
    with pytest.raises(ValueError, match="solution probabilities must lie in"):
        HHLResult(
            success_probability=0.5,
            solution_probabilities={0: probability, 1: 1.0 - probability},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=1,
            shots=8,
        )


def test_result_requires_positive_success_for_a_reported_solution():
    with pytest.raises(
        ValueError, match="a reported solution requires a positive success probability"
    ):
        HHLResult(
            success_probability=0.0,
            solution_probabilities={0: 1.0},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=1,
            shots=8,
        )


def test_result_requires_distribution_when_ancilla_succeeds():
    with pytest.raises(ValueError, match="must be reported when the ancilla succeeds"):
        HHLResult(
            success_probability=0.25,
            solution_probabilities={},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=1,
            shots=8,
        )


@pytest.mark.parametrize("success", [-0.5, 1.5, float("nan")])
def test_result_rejects_out_of_range_success_probability(success):
    with pytest.raises(ValueError, match="success_probability must lie in"):
        HHLResult(
            success_probability=success,
            solution_probabilities={},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=1,
            shots=8,
        )


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf")])
def test_result_rejects_invalid_spectral_bound(bound):
    with pytest.raises(ValueError, match="spectral_bound must be positive and finite"):
        HHLResult(
            success_probability=0.5,
            solution_probabilities={0: 1.0},
            spectral_bound=bound,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=1,
            shots=8,
        )


def test_result_distribution_is_read_only():
    result = HHLResult(
        success_probability=0.5,
        solution_probabilities={0: 0.25, 1: 0.75},
        spectral_bound=2.0,
        scaling=2.0**-2,
        num_evaluation_qubits=2,
        num_system_qubits=1,
        shots=8,
    )

    with pytest.raises(TypeError):
        result.solution_probabilities[0] = 1.0  # type: ignore[index]


def test_result_is_frozen():
    result = HHLResult(
        success_probability=0.5,
        solution_probabilities={0: 1.0},
        spectral_bound=2.0,
        scaling=2.0**-2,
        num_evaluation_qubits=2,
        num_system_qubits=1,
        shots=8,
    )

    with pytest.raises(AttributeError):
        result.shots = 16  # type: ignore[misc]


@pytest.mark.parametrize("system_width", [False, 0, -1])
def test_result_rejects_invalid_system_width(system_width):
    with pytest.raises((TypeError, ValueError), match="num_system_qubits"):
        HHLResult(
            success_probability=0.5,
            solution_probabilities={},
            spectral_bound=2.0,
            scaling=2.0**-2,
            num_evaluation_qubits=2,
            num_system_qubits=system_width,
            shots=8,
        )


def test_result_accepts_indices_wider_than_the_evaluation_register():
    # A two-qubit system under a one-qubit evaluation register still admits
    # |11>, which the evaluation width cannot express.
    result = HHLResult(
        success_probability=0.5,
        solution_probabilities={3: 1.0},
        spectral_bound=2.0,
        scaling=2.0**-1,
        num_evaluation_qubits=1,
        num_system_qubits=2,
        shots=8,
    )

    assert dict(result.solution_probabilities) == {3: 1.0}


def test_result_rejects_indices_wider_than_the_system_register():
    with pytest.raises(ValueError, match="solution indices must be valid system indices"):
        HHLResult(
            success_probability=0.5,
            solution_probabilities={4: 1.0},
            spectral_bound=2.0,
            scaling=2.0**-3,
            num_evaluation_qubits=3,
            num_system_qubits=2,
            shots=8,
        )


# ---------------------------------------------------------------------------
# HHL construction and circuit shape
# ---------------------------------------------------------------------------


def test_algorithm_rejects_non_sampler():
    with pytest.raises(TypeError, match="sampler must be a Sampler instance"):
        HHL(StatevectorSampler().run)  # type: ignore[arg-type]


def test_algorithm_is_slotted_and_exposes_injection():
    algorithm = HHL(StatevectorSampler())

    assert not hasattr(algorithm, "__dict__")
    assert isinstance(algorithm.sampler, StatevectorSampler)
    assert algorithm.components.state_preparation is AmplitudeEncoding
    with pytest.raises(AttributeError):
        algorithm.sampler = StatevectorSampler()  # type: ignore[misc]


def test_algorithm_accepts_registered_components():
    components = HHLComponents(state_preparation=AmplitudeEncoding)
    algorithm = HHL(StatevectorSampler(), components=components)

    assert algorithm.components is components


@pytest.mark.parametrize(
    ("system_width", "evaluation_width"),
    [(1, 2), (1, 6), (2, 2), (2, 5), (3, 3), (3, 6)],
)
def test_build_circuit_width_is_evaluation_plus_system_plus_ancilla(
    system_width, evaluation_width
):
    operator = np.eye(1 << system_width, dtype=np.complex128)
    problem = LinearSystemProblem(
        operator=operator,
        rhs=np.ones(1 << system_width, dtype=np.complex128),
    )

    circuit = HHL.build_circuit(
        problem,
        num_evaluation_qubits=evaluation_width,
        scaling=2.0**-evaluation_width,
    )

    assert circuit.num_qubits == evaluation_width + system_width + 1


def test_build_circuit_contains_no_classical_data():
    circuit = HHL.build_circuit(
        _diag_problem([1.0, 0.5], [1.0, 1.0]),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )

    assert not circuit.classical_vars
    assert not circuit.classical_values


def test_build_circuit_returns_fresh_independent_circuits():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])
    first = HHL.build_circuit(
        problem, num_evaluation_qubits=2, scaling=2.0**-2, bound=2.0
    )
    second = HHL.build_circuit(
        problem, num_evaluation_qubits=2, scaling=2.0**-2, bound=2.0
    )

    assert first is not second
    first.x(0)
    assert first != second


def test_build_circuit_approximation_degree_changes_the_unitary():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])
    exact = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )
    approximate = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        approximation_degree=2,
        bound=2.0,
    )

    assert not np.allclose(exact.to_matrix(), approximate.to_matrix(), atol=1e-3)


def test_build_circuit_uses_registered_component_classes():
    components = HHLComponents(
        state_preparation=_RecordingEncoding,
        phase_estimation=_RecordingPhaseEstimation,
        eigenvalue_inversion=_RecordingInversion,
    )
    _RECORDS.clear()

    HHL.build_circuit(
        _diag_problem([1.0, 0.5], [1.0, 1.0]),
        num_evaluation_qubits=2,
        scaling=2.0**-2,
        bound=2.0,
        components=components,
    )

    assert _RECORDS == [
        "state_preparation",
        "eigenvalue_inversion",
        "phase_estimation:False",
        "phase_estimation:True",
    ]


def test_build_circuit_rejects_non_problem():
    with pytest.raises(TypeError, match="problem must be a LinearSystemProblem"):
        HHL.build_circuit("problem", num_evaluation_qubits=2, scaling=2.0**-2)


# ---------------------------------------------------------------------------
# HHL.run orchestration and Sampler contract
# ---------------------------------------------------------------------------


def test_empty_batch_returns_empty_list_without_sampling():
    sampler = _ScriptedSampler([])

    assert HHL(sampler).run([]) == []
    assert sampler.calls == 0


def test_batch_results_preserve_request_order():
    sampler = _ScriptedSampler(
        [
            {_counts_key(0, 1, 1): 4},
            {_counts_key(1, 1, 1): 4},
            {_counts_key(0, 1, 1): 4},
        ]
    )

    results = HHL(sampler).run(
        [
            _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4),
            _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4),
            _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4),
        ]
    )

    assert sampler.calls == 1
    assert [sorted(result.solution_probabilities) for result in results] == [
        [0],
        [1],
        [0],
    ]


def test_complete_valid_batch_uses_one_sampler_call():
    sampler = _ScriptedSampler([{_counts_key(0, 1, 1): 4}] * 4)
    requests = [
        _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4) for _ in range(4)
    ]

    results = HHL(sampler).run(requests)

    assert sampler.calls == 1
    assert len(results) == 4


def test_complete_batch_is_validated_before_sampler_call():
    sampler = _ScriptedSampler([{_counts_key(0, 1, 1): 4}])

    with pytest.raises(TypeError, match="requests must contain HHLRequest instances"):
        HHL(sampler).run(
            [
                _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4),
                "invalid",  # type: ignore[list-item]
            ]
        )

    assert sampler.calls == 0


def test_run_rejects_non_list_sampler_result_container():
    class _TupleSampler(Sampler):
        def run(self, requests):
            return tuple(
                ExecutionResult.from_counts(
                    "tuple",
                    list(request.qubits),
                    request.shots,
                    len(request.qubits),
                    {"0" * len(request.qubits): request.shots},
                    "fake-backend",
                )
                for request in requests
            )

    with pytest.raises(TypeError, match="sampler must return a list"):
        HHL(_TupleSampler()).run(
            [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4)]
        )


def test_run_rejects_non_execution_results():
    class _DictSampler(Sampler):
        def run(self, requests):
            return [{"0" * len(request.qubits): request.shots} for request in requests]

    with pytest.raises(TypeError, match="cqlib.device.ExecutionResult"):
        HHL(_DictSampler()).run(
            [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4)]
        )


def test_run_rejects_wrong_result_count():
    sampler = _ScriptedSampler([{_counts_key(0, 1, 1): 4}])

    with pytest.raises(ValueError, match="one result per request"):
        HHL(sampler).run(
            [
                _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4),
                _request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4),
            ]
        )


def test_run_rejects_mismatched_result_shots():
    class _ShiftedShotsSampler(Sampler):
        def run(self, requests):
            return [
                ExecutionResult.from_counts(
                    "shifted",
                    list(request.qubits),
                    request.shots + 1,
                    len(request.qubits),
                    {"0" * len(request.qubits): request.shots + 1},
                    "fake-backend",
                )
                for request in requests
            ]

    with pytest.raises(ValueError, match="shots must match"):
        HHL(_ShiftedShotsSampler()).run(
            [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4)]
        )


@pytest.mark.parametrize("mode", ["queued", "failed"])
def test_run_rejects_unsuccessful_or_nonterminal_results(mode):
    with pytest.raises(ValueError, match="successful terminal"):
        HHL(_InvalidResultSampler(mode)).run(
            [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4)]
        )


def test_run_rejects_result_width_mismatch():
    with pytest.raises(ValueError, match="width"):
        HHL(_InvalidResultSampler("width")).run(
            [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4)]
        )


def test_run_rejects_result_qubit_order_mismatch():
    with pytest.raises(ValueError, match="qubits"):
        HHL(_InvalidResultSampler("qubits")).run(
            [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=4)]
        )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"0": 4}, "width"),
        ({"0x": 4}, "binary"),
        ({"00": True, "11": 3}, "non-negative integers"),
        ({"00": 1.5, "11": 2.5}, "non-negative integers"),
        ({"00": -1, "11": 5}, "non-negative integers"),
    ],
)
def test_run_rejects_malformed_count_entries(counts, message):
    circuit = HHL.build_circuit(
        _diag_problem([1.0, 0.5], [1.0, 1.0]),
        num_evaluation_qubits=2,
        scaling=2.0**-2,
        bound=2.0,
    )
    # The measured register is the system plus the ancilla, excluding the
    # evaluation register, so the requested width is two for this problem.
    request = SamplerRequest(circuit, tuple(circuit.qubits[2:]), 4)
    valid = ExecutionResult.from_counts(
        "valid",
        list(request.qubits),
        request.shots,
        len(request.qubits),
        {"00": request.shots},
        "fake-backend",
    )
    malformed = SimpleNamespace(
        status=valid.status,
        shots=valid.shots,
        num_qubits=valid.num_qubits,
        qubits=valid.qubits,
        counts=counts,
    )

    with pytest.raises(ValueError, match=message):
        validate_sampler_result(malformed, request)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Count reduction
# ---------------------------------------------------------------------------


def test_reduction_selects_only_ancilla_one_shots():
    sampler = _ScriptedSampler(
        [{"10": 3, "11": 5, _counts_key(0, 0, 1): 2}]
    )

    result = HHL(sampler).run(
        [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=10)]
    )[0]

    assert result.success_probability == pytest.approx(0.8)
    assert dict(result.solution_probabilities) == pytest.approx({0: 0.375, 1: 0.625})


def test_reduction_reports_empty_distribution_without_ancilla_success():
    sampler = _ScriptedSampler([{_counts_key(0, 0, 1): 6, _counts_key(1, 0, 1): 2}])

    result = HHL(sampler).run(
        [_request(_diag_problem([1.0, 0.5], [1.0, 1.0]), shots=8)]
    )[0]

    assert result.success_probability == 0.0
    assert dict(result.solution_probabilities) == {}


def test_reduction_reports_metadata_from_the_request():
    sampler = _ScriptedSampler([{_counts_key(0, 1, 2): 16}])
    problem = _diag_problem([1.0, 0.5, 0.25, 0.125], [1.0, 1.0, 1.0, 1.0])
    request = _request(problem, num_evaluation_qubits=3, scaling=2.0**-3, shots=16, bound=2.0)

    result = HHL(sampler).run([request])[0]

    assert result.spectral_bound == 2.0
    assert result.scaling == 2.0**-3
    assert result.num_evaluation_qubits == 3
    assert result.shots == 16


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------


@pytest.mark.reference
def test_diagonal_two_by_two_matches_analytic_solution():
    operator = np.diag([1.0, 0.5])
    rhs = np.array([1.0, 1.0])
    circuit = HHL.build_circuit(
        _diag_problem([1.0, 0.5], rhs),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )

    np.testing.assert_allclose(
        _exact_distribution(circuit, _EVALUATION_QUBITS, 1),
        _analytic_solution(operator, rhs),
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_diagonal_four_by_four_matches_analytic_solution():
    eigenvalues = [1.0, 0.5, 0.25, 0.125]
    rhs = np.ones(4)
    circuit = HHL.build_circuit(
        _diag_problem(eigenvalues, rhs),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )

    np.testing.assert_allclose(
        _exact_distribution(circuit, _EVALUATION_QUBITS, 2),
        _analytic_solution(np.diag(eigenvalues), rhs),
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_success_probability_matches_closed_form():
    eigenvalues = [1.0, 0.5, 0.25, 0.125]
    rhs = np.ones(4)
    scaling = 2.0**-_EVALUATION_QUBITS
    circuit = HHL.build_circuit(
        _diag_problem(eigenvalues, rhs),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=scaling,
        bound=2.0,
    )
    joint = _joint_distribution(circuit, _EVALUATION_QUBITS, 2)

    assert _success_probability(joint) == pytest.approx(
        _analytic_success(np.diag(eigenvalues), rhs, 2.0, scaling),
        abs=_SCALING_TOLERANCE,
    )
    assert _success_probability(joint) == pytest.approx(85.0 / 256.0, abs=_SCALING_TOLERANCE)


@pytest.mark.reference
def test_scaling_quadratically_changes_success_probability():
    eigenvalues = [1.0, 0.5, 0.25, 0.125]
    rhs = np.ones(4)
    small = HHL.build_circuit(
        _diag_problem(eigenvalues, rhs),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )
    smaller = HHL.build_circuit(
        _diag_problem(eigenvalues, rhs),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0 ** -(_EVALUATION_QUBITS + 1),
        bound=2.0,
    )
    small_success = _success_probability(
        _joint_distribution(small, _EVALUATION_QUBITS, 2)
    )
    smaller_success = _success_probability(
        _joint_distribution(smaller, _EVALUATION_QUBITS, 2)
    )

    assert small_success / smaller_success == pytest.approx(4.0, rel=1e-9)


@pytest.mark.reference
def test_rotated_positive_definite_matrix_matches_analytic_solution():
    angle = 0.6
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    operator = rotation @ np.diag([1.0, 0.5]) @ rotation.T
    rhs = np.array([1.0, 0.0])
    problem = LinearSystemProblem(operator=operator, rhs=rhs)
    circuit = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )

    np.testing.assert_allclose(
        _exact_distribution(circuit, _EVALUATION_QUBITS, 1),
        _analytic_solution(operator, rhs),
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_eigenstate_rhs_concentrates_on_its_eigenvector():
    angle = 0.6
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    operator = rotation @ np.diag([1.0, 0.5]) @ rotation.T
    eigenvector = rotation[:, 1]
    problem = LinearSystemProblem(operator=operator, rhs=eigenvector)
    circuit = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )

    np.testing.assert_allclose(
        _exact_distribution(circuit, _EVALUATION_QUBITS, 1),
        np.abs(eigenvector) ** 2,
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_reference_joint_distribution_matches_direct_statevector_readout():
    eigenvalues = [1.0, 0.5, 0.25, 0.125]
    rhs = np.ones(4)
    scaling = 2.0**-_EVALUATION_QUBITS
    circuit = HHL.build_circuit(
        _diag_problem(eigenvalues, rhs),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=scaling,
        bound=2.0,
    )

    joint = _joint_distribution(circuit, _EVALUATION_QUBITS, 2)
    reference = _analytic_joint(
        np.diag(eigenvalues),
        rhs,
        2.0,
        scaling,
        _EVALUATION_QUBITS,
    )
    for key, probability in reference.items():
        assert joint.get(key, 0.0) == pytest.approx(probability, abs=_SCALING_TOLERANCE)


@pytest.mark.reference
def test_default_bound_retains_every_eigenvalue_branch():
    # With alpha = lambda_max / (1 - 2**-m) = 16/15 the largest eigenvalue
    # lands on grid point 15 and the smallest on grid point 5, so the
    # reciprocal encoding is exact for both branches.
    alpha = 16.0 / 15.0
    scaling = 2.0**-_EVALUATION_QUBITS
    eigenvalues = [1.0, 1.0 / 3.0]
    problem = _diag_problem(eigenvalues, [1.0, 1.0])
    circuit = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=scaling,
    )
    joint = _joint_distribution(circuit, _EVALUATION_QUBITS, 1)

    # A x = b gives x proportional to (1, 3), hence 1/10 and 9/10.
    np.testing.assert_allclose(
        _dense_distribution(joint, 1),
        [0.1, 0.9],
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )
    assert _success_probability(joint) == pytest.approx(
        _analytic_success(np.diag(eigenvalues), [1.0, 1.0], alpha, scaling),
        abs=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_bound_at_the_largest_eigenvalue_warns_and_drops_that_branch():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])
    with pytest.warns(UserWarning, match="rounds onto the zero grid point"):
        aliased_bound = HHL.build_circuit(
            problem,
            num_evaluation_qubits=_EVALUATION_QUBITS,
            scaling=2.0**-_EVALUATION_QUBITS,
            bound=1.0,
        )
    raised_bound = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )
    aliased_joint = _joint_distribution(aliased_bound, _EVALUATION_QUBITS, 1)
    raised_joint = _joint_distribution(raised_bound, _EVALUATION_QUBITS, 1)

    # The largest eigenvalue aliases onto the phase-zero grid point, whose
    # reciprocal amplitude is zero, so only the smaller branch survives.
    np.testing.assert_allclose(
        _dense_distribution(aliased_joint, 1),
        [0.0, 1.0],
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )
    assert _success_probability(aliased_joint) == pytest.approx(
        0.5 * (2.0**-_EVALUATION_QUBITS) ** 2 / 0.5**2,
        abs=_SCALING_TOLERANCE,
    )
    np.testing.assert_allclose(
        _dense_distribution(raised_joint, 1),
        [0.2, 0.8],
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_hhl_circuit_is_unitary_and_uncomputes_the_evaluation_register():
    circuit = HHL.build_circuit(
        _diag_problem([1.0, 0.5], [1.0, 1.0]),
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )
    matrix = np.asarray(circuit.to_matrix(), dtype=np.complex128)
    dimension = matrix.shape[0]
    state = matrix[:, 0]
    evaluation_bits = np.arange(state.size) & ((1 << _EVALUATION_QUBITS) - 1)

    np.testing.assert_allclose(
        matrix.conj().T @ matrix,
        np.eye(dimension),
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )
    assert float(np.sum(np.abs(state[evaluation_bits == 0]) ** 2)) == pytest.approx(
        1.0, abs=_SCALING_TOLERANCE
    )


@pytest.mark.reference
def test_paper_example_matches_the_classical_solution():
    operator = np.array([[1.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0]])
    rhs = np.array([1.0, 0.0])
    problem = LinearSystemProblem(operator=operator, rhs=rhs)
    # The bound 8/3 normalizes the eigenvalues 4/3 and 2/3 onto the 16-point grid.
    circuit = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=8.0 / 3.0,
    )

    np.testing.assert_allclose(
        _exact_distribution(circuit, _EVALUATION_QUBITS, 1),
        _analytic_solution(operator, rhs),
        rtol=0.0,
        atol=_SCALING_TOLERANCE,
    )


@pytest.mark.reference
def test_pauli_operator_protocol_leaves_the_ancilla_untouched():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])
    circuit = HHL.build_circuit(
        problem,
        num_evaluation_qubits=_EVALUATION_QUBITS,
        scaling=2.0**-_EVALUATION_QUBITS,
        bound=2.0,
    )
    joint = _joint_distribution(circuit, _EVALUATION_QUBITS, 1)

    assert set(joint) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for system in (0, 1):
        assert joint[(system, 0)] > 0.0
        assert joint[(system, 1)] > 0.0


# ---------------------------------------------------------------------------
# End-to-end execution paths
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_seed_reproduces_the_sampled_distribution():
    problem = _diag_problem([1.0, 0.5], [1.0, 1.0])
    algorithm = HHL(StatevectorSampler())
    requests = [
        _request(problem, shots=4096, bound=2.0, seed=7),
        _request(problem, shots=4096, bound=2.0, seed=7),
    ]

    first, second = algorithm.run(requests)

    assert first.success_probability == second.success_probability
    assert dict(first.solution_probabilities) == dict(second.solution_probabilities)


@pytest.mark.integration
def test_statevector_sampling_is_close_to_the_exact_distribution():
    problem = _diag_problem([1.0, 0.5, 0.25, 0.125], [1.0, 1.0, 1.0, 1.0])
    result = HHL(StatevectorSampler()).run(
        [_request(problem, shots=20000, bound=2.0, seed=11)]
    )[0]

    expected = _analytic_solution(np.diag([1.0, 0.5, 0.25, 0.125]), np.ones(4))
    observed = np.zeros(4)
    for index, probability in result.solution_probabilities.items():
        observed[index] = probability

    assert result.success_probability == pytest.approx(85.0 / 256.0, abs=0.02)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.03)


@pytest.mark.integration
def test_circuit_rhs_matches_the_equivalent_vector_rhs():
    operator = np.diag([1.0, 0.5])
    preparation = Circuit(1)
    preparation.h(0)
    circuit_problem = LinearSystemProblem(operator=operator, rhs=preparation)
    vector_problem = _diag_problem([1.0, 0.5], [1.0, 1.0])

    from_circuit = HHL(StatevectorSampler()).run(
        [_request(circuit_problem, shots=4096, bound=2.0, seed=3)]
    )[0]
    from_vector = HHL(StatevectorSampler()).run(
        [_request(vector_problem, shots=4096, bound=2.0, seed=3)]
    )[0]

    # The two preparations inject the same |+> state, so their sampled
    # distributions agree within shot noise.
    assert from_circuit.success_probability == pytest.approx(
        from_vector.success_probability, abs=0.02
    )
    assert dict(from_circuit.solution_probabilities) == pytest.approx(
        dict(from_vector.solution_probabilities), abs=0.02
    )
    _assert_solution_matches_analytic(from_circuit, operator, [1.0, 1.0], 4096)


@pytest.mark.integration
def test_hamiltonian_operator_matches_the_equivalent_dense_matrix():
    hamiltonian = Hamiltonian(1)
    hamiltonian.add_term(PauliString.from_str("I"), 1.5)
    hamiltonian.add_term(PauliString.from_str("Z"), 0.5)
    problem = LinearSystemProblem(operator=hamiltonian, rhs=[1.0, 1.0])
    dense = _diag_problem([2.0, 1.0], [1.0, 1.0])
    algorithm = HHL(StatevectorSampler())

    from_hamiltonian, from_dense = algorithm.run(
        [
            _request(problem, shots=4096, bound=4.0, seed=5),
            _request(dense, shots=4096, bound=4.0, seed=5),
        ]
    )

    assert from_hamiltonian.success_probability == pytest.approx(
        from_dense.success_probability, abs=0.02
    )
    assert dict(from_hamiltonian.solution_probabilities) == pytest.approx(
        dict(from_dense.solution_probabilities), abs=0.03
    )
    _assert_solution_matches_analytic(
        from_hamiltonian, np.diag([2.0, 1.0]), [1.0, 1.0], 4096
    )


# ---------------------------------------------------------------------------
# Numerical invariants
# ---------------------------------------------------------------------------


def test_num_unitary_applications_depends_only_on_the_evaluation_width():
    wide_system = HHLResult(
        success_probability=0.5,
        solution_probabilities={0: 1.0},
        spectral_bound=2.0,
        scaling=2.0**-3,
        num_evaluation_qubits=3,
        num_system_qubits=5,
        shots=8,
    )
    many_shots = HHLResult(
        success_probability=0.5,
        solution_probabilities={0: 1.0},
        spectral_bound=1024.0,
        scaling=2.0**-3,
        num_evaluation_qubits=3,
        num_system_qubits=2,
        shots=1 << 20,
    )

    assert wide_system.num_unitary_applications == 14
    assert many_shots.num_unitary_applications == 14


def test_sampled_distributions_are_float64_precision():
    sampler = _ScriptedSampler([{_counts_key(0, 1, 2): 1, _counts_key(3, 1, 2): 3}])
    problem = _diag_problem([1.0, 0.5, 0.25, 0.125], [1.0, 1.0, 1.0, 1.0])

    result = HHL(sampler).run(
        [_request(problem, num_evaluation_qubits=3, scaling=2.0**-3, shots=4)]
    )[0]

    assert isinstance(result.success_probability, float)
    values = np.asarray(list(result.solution_probabilities.values()))
    assert values.dtype == np.float64
    for index, probability in result.solution_probabilities.items():
        assert isinstance(index, int)
        assert isinstance(probability, float)
