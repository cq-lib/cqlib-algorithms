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

"""Linear-system solving with the classic phase-estimation HHL algorithm.

The Harrow-Hassidim-Lloyd solver prepares, in an ``n``-qubit system register,
a state proportional to ``A**-1 |b>`` for a Hermitian positive-definite
operator ``A`` and a right-hand side ``b``. The state is produced only
conditioned on the ``|1>`` outcome of a single ancilla qubit.

The circuit applies four blocks in order: preparation of ``|b>``, phase
estimation of ``U = exp(2*pi*i*A/alpha)``, a controlled reciprocal rotation,
and the inverse phase estimation. The reported solution is the post-selected
system-register distribution; the algorithm does not classically reconstruct
the full vector ``x``.

References
----------
[1] A. W. Harrow, A. Hassidim, and S. Lloyd, "Quantum algorithm for linear
    systems of equations," Physical Review Letters, 103, 150502, 2009.
    DOI: 10.1103/PhysRevLett.103.150502.
[2] D. W. Berry, "High-order quantum algorithm for solving linear
    differential equations," Journal of Physics A, 47, 105301, 2014.
    DOI: 10.1088/1751-8113/47/10/105301.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast, final, overload

import numpy as np
import numpy.typing as npt
from cqlib import Circuit
from cqlib.qis import Hamiltonian, TrotterMode

from .._utils.hamiltonian import snapshot_nonzero_hermitian_hamiltonian
from .._utils.numeric import normalize_positive_integer, validate_optional_seed
from .._utils.phase_estimation import (
    circuit_from_unitary_matrix,
    normalize_evaluation_qubits,
    normalize_shots,
    snapshot_phase_preparation,
    validate_static_unitary_circuit,
)
from ..operators import (
    AmplitudeEncoding,
    PhaseEstimationCircuit,
    UserCircuitPreparation,
)
from ..operators.arithmetic import ExactReciprocal
from ..operators.state_preparation._state_synthesis import (
    copy_complex_vector,
    normalize_nonzero,
)
from ..protocols.sampler import Sampler, SamplerRequest, require_sampler_results
from ..protocols.state_preparation import StatePreparation

_HERMITIAN_RTOL = 1e-8
_HERMITIAN_ATOL = 1e-10
_BOUND_RTOL = 1e-9
_UNITARY_LABEL = "hhl-unitary"


def _validate_scaling(scaling: object, *, num_evaluation_qubits: int) -> None:
    """Validate the reciprocal amplitude scaling ``0 < s <= 2**-m``."""
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        raise TypeError("scaling must be a real number")
    resolved = float(scaling)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("scaling must be positive and finite")
    if resolved > 2.0**-num_evaluation_qubits:
        raise ValueError("scaling must not exceed 2**-num_evaluation_qubits")


def _validate_approximation_degree(
    value: object, *, num_evaluation_qubits: int
) -> None:
    """Validate a small-angle inverse-QFT approximation degree."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("approximation_degree must be an integer")
    if not 0 <= value < num_evaluation_qubits:
        raise ValueError(
            "approximation_degree must satisfy "
            "0 <= approximation_degree < num_evaluation_qubits"
        )


def _dense_matrix(operator: object) -> npt.NDArray[np.complex128]:
    """Return the dense ``complex128`` matrix of an operator"""
    if isinstance(operator, np.ndarray):
        return np.asarray(operator, dtype=np.complex128)
    if isinstance(operator, Hamiltonian):
        return np.asarray(operator.to_matrix(), dtype=np.complex128)
    raise TypeError("operator must be a numpy.ndarray or a Hamiltonian")


def _validate_matrix(matrix: npt.NDArray[np.complex128]) -> None:
    """Validate that a dense operator matrix is square, finite, and invertible."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    dimension = matrix.shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("operator dimension must be a power of two above one")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator must be finite")
    if np.linalg.matrix_rank(matrix) != dimension:
        raise ValueError("operator must be non-singular")


def _operator_matrix(operator: object) -> npt.NDArray[np.complex128]:
    """Return the validated dense ``complex128`` matrix of an operator."""
    matrix = _dense_matrix(operator)
    _validate_matrix(matrix)
    return matrix


def _operator_width(operator: object) -> int:
    """Return the system-register width of a validated operator."""
    if isinstance(operator, np.ndarray):
        return int(operator.shape[0]).bit_length() - 1
    if isinstance(operator, Hamiltonian):
        return int(operator.num_qubits)
    raise TypeError("operator must be a numpy.ndarray or a Hamiltonian")


def _validate_operator(operator: object) -> None:
    """Validate the structural invariants of the linear operator."""
    _validate_matrix(_dense_matrix(operator))


def _snapshot_operator(
    operator: object,
) -> npt.NDArray[np.complex128] | Hamiltonian:
    """Return a validated, independent snapshot of the linear operator."""
    if isinstance(operator, Hamiltonian):
        _validate_operator(operator)
        return operator.copy()
    snapshot: npt.NDArray[np.complex128] = _operator_matrix(operator).copy()
    return snapshot


def _rhs_amplitudes(rhs: object) -> tuple[complex, ...]:
    """Return a finite, non-zero, power-of-two complex copy of a vector."""
    if isinstance(rhs, np.ndarray):
        values: Sequence[complex] = rhs.tolist()
    elif isinstance(rhs, (str, bytes)) or not isinstance(rhs, Sequence):
        raise TypeError("rhs must be a sequence or a StatePreparation")
    else:
        values = rhs

    copied = copy_complex_vector(values, name="rhs")
    normalize_nonzero(copied, name="rhs")
    return copied


def _validate_rhs_width(rhs: Sequence[complex] | Circuit, *, num_qubits: int) -> None:
    """Check the vector length or preparation width against the system register."""
    if isinstance(rhs, Circuit):
        if rhs.num_qubits != num_qubits:
            raise ValueError("rhs circuit width must equal the system width")
    elif len(rhs) != 1 << num_qubits:
        raise ValueError("rhs must have length 2**num_qubits")


def _hermitian_spectrum(operator: object) -> npt.NDArray[np.float64]:
    """Return the ascending real spectrum of a Hermitian operator."""
    matrix = _operator_matrix(operator)
    if not np.allclose(
        matrix,
        matrix.conj().T,
        rtol=_HERMITIAN_RTOL,
        atol=_HERMITIAN_ATOL,
    ):
        raise ValueError("operator must be Hermitian")
    return np.asarray(np.linalg.eigvalsh(matrix), dtype=np.float64)


def _spectral_bound(
    operator: object,
    bound: float | None,
    *,
    num_evaluation_qubits: int,
) -> float:
    r"""Resolve the spectral bound ``alpha`` of a positive-definite operator.

    The reciprocal encoding is exact only where ``lambda / alpha`` lands on an
    integer multiple of the phase-grid step :math:`2^{-m}`; the zero grid point
    encodes the zero eigenvalue and applies no rotation, so any branch that
    rounds onto it is dropped. The default bound therefore shrinks the largest
    eigenvalue by one grid step, ``alpha = lambda_max / (1 - 2**-m)``, which
    places ``lambda_max / alpha`` on the last resolvable grid point instead of
    the period endpoint.

    Args:
        operator: Dense matrix or Hamiltonian defining ``A``.
        bound: Requested spectral bound, or ``None`` to derive
            ``lambda_max / (1 - 2**-m)`` from the spectrum.
        num_evaluation_qubits: Width ``m`` of the evaluation register, which
            fixes the phase-grid step ``2**-m``.

    Returns:
        The resolved spectral bound ``alpha``.

    Raises:
        TypeError: If ``operator`` or ``bound`` has the wrong type.
        ValueError: If ``A`` is not Hermitian positive definite, or ``bound``
            is not positive, finite, and at least the largest eigenvalue.

    Warns:
        UserWarning: If an eigenvalue is closer to the period endpoint than
            half a phase-grid step, so it rounds onto the zero grid point and
            the reciprocal encoding drops it.
    """
    spectrum = _hermitian_spectrum(operator)
    smallest = float(spectrum[0])
    if smallest <= 0.0:
        raise ValueError("operator must be positive definite")
    largest = float(spectrum[-1])
    if bound is None:
        alpha = largest / (1.0 - 2.0**-num_evaluation_qubits)
    else:
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise TypeError("bound must be a real number or None")
        alpha = float(bound)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("bound must be positive and finite")
        if alpha < largest and not math.isclose(
            alpha,
            largest,
            rel_tol=_BOUND_RTOL,
            abs_tol=0.0,
        ):
            raise ValueError("bound must not be smaller than the largest eigenvalue")

    # A branch survives only while its phase rounds to a non-zero grid point.
    # Phase 0 and phase 1 address the same zero grid point, so either end can
    # swallow a branch that is closer to the endpoint than half a grid step.
    # The distances are compared on the eigenvalue scale, where the default
    # bound sits exactly on the boundary of the upper test, and the relative
    # tolerance keeps the round trip through ``alpha`` from reporting that
    # exact case.
    half_step = 2.0 ** -(num_evaluation_qubits + 1)
    low_cutoff = alpha * half_step
    high_cutoff = alpha * (1.0 - half_step)
    if (
        smallest < low_cutoff
        and not math.isclose(smallest, low_cutoff, rel_tol=_BOUND_RTOL, abs_tol=0.0)
    ) or (
        largest > high_cutoff
        and not math.isclose(largest, high_cutoff, rel_tol=_BOUND_RTOL, abs_tol=0.0)
    ):
        warnings.warn(
            "the spectral bound leaves an eigenvalue closer to the period "
            "endpoint than half a phase-grid step; that branch rounds onto the "
            "zero grid point and is dropped by the reciprocal encoding, so "
            "raise bound or enlarge num_evaluation_qubits",
            UserWarning,
            stacklevel=2,
        )
    return alpha


def _hermitian_unitary(operator: object, alpha: float) -> Circuit:
    """Build the circuit of ``U = exp(2*pi*i*A/alpha)``.

    A dense operator is diagonalized with a Hermitian eigendecomposition and
    wrapped in a numeric unitary gate. A Hamiltonian is scaled and evolved
    with a product formula instead, because its metric is not kept dense.

    Args:
        operator: Dense matrix or Hamiltonian defining ``A``.
        alpha: Positive spectral bound used to normalize the spectrum.

    Returns:
        A circuit whose unitary is ``exp(2*pi*i*A/alpha)``.

    Raises:
        TypeError: If ``operator`` has the wrong type.
        ValueError: If the operator or the spectral bound is invalid.
    """
    if isinstance(operator, np.ndarray):
        matrix = np.asarray(operator, dtype=np.complex128)
        values, vectors = np.linalg.eigh(matrix)
        phases = np.exp(2.0j * math.pi * values / alpha)
        unitary = np.asarray(
            (vectors * phases) @ vectors.conj().T,
            dtype=np.complex128,
        )
        return circuit_from_unitary_matrix(unitary, _UNITARY_LABEL)
    if isinstance(operator, Hamiltonian):
        snapshot = snapshot_nonzero_hermitian_hamiltonian(operator, name="operator")
        snapshot.scale(1.0 / alpha)
        # Core synthesizes exp(-i*t*H); t = -2*pi yields exp(+2*pi*i*H).
        return snapshot.to_evolution_circuit(
            -2.0 * math.pi,
            1,
            TrotterMode.second_order(),
        )
    raise TypeError("operator must be a numpy.ndarray or a Hamiltonian")


@dataclass(frozen=True, slots=True, init=False)
class LinearSystemProblem:
    """Semantic definition of the linear system ``A x = b``.

    The container enforces structural validity only, so it can be shared with
    solvers of linear differential equations whose discretization reduces to
    the same algebraic form [2]. Algorithm-specific operator constraints
    (HHL's Hermitian positive-definiteness) are attached by the algorithm.

    ``operator`` is normalized to ``complex128`` and ``rhs`` is copied, so the
    stored fields are an independent snapshot. A preparation component is
    built and validated as a fully bound static unitary circuit of width
    ``n`` preparing ``|b>`` from ``|0>``.

    Args:
        operator: Finite, square, non-singular operator ``A`` of dimension
            ``2**n``, or a Hamiltonian.
        rhs: Finite nonzero vector of length ``2**n``, or a StatePreparation
            preparing ``|b>`` on ``n`` qubits.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the operator is non-finite, non-square, not of
            dimension ``2**n``, or singular, or the right-hand side has the
            wrong length, is zero, or is not a static unitary circuit.
    """

    operator: npt.NDArray[np.complex128] | Hamiltonian
    rhs: Sequence[complex] | StatePreparation
    _rhs_circuit: Circuit | None = field(init=False, repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        operator: npt.NDArray[np.generic] | Hamiltonian,
        rhs: Sequence[complex] | npt.NDArray[np.generic] | StatePreparation,
    ) -> None:
        operator = _snapshot_operator(operator)
        stored_rhs = (
            snapshot_phase_preparation(rhs, "rhs")
            if isinstance(rhs, StatePreparation)
            else _rhs_amplitudes(rhs)
        )
        _validate_rhs_width(stored_rhs, num_qubits=_operator_width(operator))
        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "rhs",
            UserCircuitPreparation(stored_rhs)
            if isinstance(stored_rhs, Circuit)
            else stored_rhs,
        )
        object.__setattr__(
            self,
            "_rhs_circuit",
            stored_rhs if isinstance(stored_rhs, Circuit) else None,
        )

    @property
    def rhs_circuit(self) -> Circuit | None:
        """Return the stored preparation snapshot, or None for vector input."""
        return self._rhs_circuit

    @property
    def num_qubits(self) -> int:
        """Return the system-register width ``n``."""
        return _operator_width(self.operator)

    def validate(self) -> None:
        """Revalidate the shallowly frozen problem before execution."""
        _validate_operator(self.operator)
        if isinstance(self.rhs, StatePreparation):
            if self._rhs_circuit is None:
                raise ValueError("rhs preparation snapshot is missing")
            validate_static_unitary_circuit(self._rhs_circuit, "rhs")
            _validate_rhs_width(self._rhs_circuit, num_qubits=self.num_qubits)
        else:
            _validate_rhs_width(self.rhs, num_qubits=self.num_qubits)


@dataclass(frozen=True, slots=True)
class HHLComponents:
    """Concrete operator classes instantiated while assembling the circuit.

    Each field names a class, not an instance: the assembly instantiates it
    with the per-request parameters, so substituting a subclass configures the
    same task with a different operator implementation. The ``A -> U``
    adaptation is intrinsic to the algorithm and is not a component.

    Args:
        state_preparation: StatePreparation subclass whose constructor accepts
            the right-hand-side feature vector. Used when the problem supplies a
            vector; a supplied preparation component is used directly.
        phase_estimation: Phase-estimation class used for the forward and the
            adjoint block.
        eigenvalue_inversion: Controlled reciprocal-rotation class.
    """

    state_preparation: type[StatePreparation] = AmplitudeEncoding
    phase_estimation: type[PhaseEstimationCircuit] = PhaseEstimationCircuit
    eigenvalue_inversion: type[ExactReciprocal] = ExactReciprocal

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _validate_components(self)


def _validate_components(components: object) -> None:
    """Validate the registered operator classes and their roles."""
    if not isinstance(components, HHLComponents):
        raise TypeError("components must be an HHLComponents instance")
    if not (
        isinstance(components.state_preparation, type)
        and issubclass(components.state_preparation, StatePreparation)
    ):
        raise TypeError("state_preparation must be a StatePreparation subclass")
    if not (
        isinstance(components.phase_estimation, type)
        and issubclass(components.phase_estimation, PhaseEstimationCircuit)
    ):
        raise TypeError("phase_estimation must be a PhaseEstimationCircuit subclass")
    if not (
        isinstance(components.eigenvalue_inversion, type)
        and issubclass(components.eigenvalue_inversion, ExactReciprocal)
    ):
        raise TypeError("eigenvalue_inversion must be an ExactReciprocal subclass")


@dataclass(frozen=True, slots=True)
class HHLRequest:
    """Configure one HHL run for the linear system ``A x = b``.

    ``alpha`` is resolved from ``bound`` when it is given, and otherwise from
    the largest eigenvalue, shrunk by one phase-grid step so that it stays
    resolvable. Because ``alpha`` fixes the phase scale, ``bound`` must not be
    smaller than that eigenvalue. A ``bound`` that pushes an eigenvalue closer
    to the period endpoint than half a grid step rounds that branch onto the
    zero grid point, where the reciprocal encoding drops it; a ``UserWarning``
    is raised in that case.

    Args:
        problem: Hermitian positive-definite linear system.
        num_evaluation_qubits: Width ``m`` of the evaluation register.
        scaling: Reciprocal amplitude scaling ``s`` with ``0 < s <= 2**-m``.
        shots: Positive number of circuit samples.
        approximation_degree: Small-angle inverse-QFT layers to omit.
        bound: Positive spectral bound ``alpha``, or ``None`` for the largest
            eigenvalue shrunk by one phase-grid step.
        seed: Non-negative seed of the shot draw, or ``None``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a setting is out of range, the problem invariant fails,
            or the operator is not Hermitian positive definite.

    Warns:
        UserWarning: If the resolved bound leaves an eigenvalue within one
            phase-grid step of the period endpoint.
    """

    problem: LinearSystemProblem
    num_evaluation_qubits: int
    scaling: float
    shots: int
    approximation_degree: int = 0
    bound: float | None = None
    seed: int | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the request and its operator constraints."""
        if not isinstance(self.problem, LinearSystemProblem):
            raise TypeError("problem must be a LinearSystemProblem")
        self.problem.validate()
        normalize_evaluation_qubits(self.num_evaluation_qubits)
        _validate_scaling(
            self.scaling,
            num_evaluation_qubits=self.num_evaluation_qubits,
        )
        normalize_shots(self.shots)
        validate_optional_seed(self.seed)
        _validate_approximation_degree(
            self.approximation_degree,
            num_evaluation_qubits=self.num_evaluation_qubits,
        )
        _spectral_bound(
            self.problem.operator,
            self.bound,
            num_evaluation_qubits=self.num_evaluation_qubits,
        )


@dataclass(frozen=True, slots=True)
class HHLResult:
    """Post-selected HHL outcome for one request.

    ``solution_probabilities`` is sparse over observed system-register
    integers in little-endian basis order, so index ``i`` is the basis state
    ``|i>`` of the solution register. It is empty when no shot lands in the
    ``ancilla = 1`` branch, and otherwise renormalized to sum to one.
    ``success_probability`` is the raw fraction of shots whose ancilla outcome
    is ``1``, before that renormalization.

    The evaluation-register width and the system-register width are
    independent settings, so ``num_system_qubits`` is stored separately and
    fixes the valid range of ``solution_probabilities``.

    Args:
        success_probability: Fraction of shots with ancilla outcome ``1``.
        solution_probabilities: Post-selected system-register distribution.
        spectral_bound: Spectral bound ``alpha`` used for the phase scale.
        scaling: Reciprocal amplitude scaling ``s``.
        num_evaluation_qubits: Width ``m`` of the evaluation register.
        num_system_qubits: Width ``n`` of the system register.
        shots: Samples used for the circuit.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the counts, probabilities, or widths are inconsistent.
    """

    success_probability: float
    solution_probabilities: Mapping[int, float]
    spectral_bound: float
    scaling: float
    num_evaluation_qubits: int
    num_system_qubits: int
    shots: int

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        width = normalize_evaluation_qubits(self.num_evaluation_qubits)
        system_width = normalize_positive_integer(
            self.num_system_qubits,
            "num_system_qubits",
        )
        normalize_shots(self.shots)
        _validate_scaling(self.scaling, num_evaluation_qubits=width)
        if (
            isinstance(self.spectral_bound, bool)
            or not isinstance(self.spectral_bound, (int, float))
            or not math.isfinite(float(self.spectral_bound))
            or float(self.spectral_bound) <= 0.0
        ):
            raise ValueError("spectral_bound must be positive and finite")
        if (
            isinstance(self.success_probability, bool)
            or not isinstance(self.success_probability, (int, float))
            or not math.isfinite(float(self.success_probability))
            or not 0.0 <= float(self.success_probability) <= 1.0
        ):
            raise ValueError("success_probability must lie in [0, 1]")
        if not isinstance(self.solution_probabilities, Mapping):
            raise TypeError("solution_probabilities must be a mapping")

        probabilities = dict(self.solution_probabilities)
        for index, probability in probabilities.items():
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < (1 << system_width)
            ):
                raise ValueError("solution indices must be valid system indices")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or not 0.0 < float(probability) <= 1.0
            ):
                raise ValueError("solution probabilities must lie in (0, 1]")

        if probabilities:
            if not math.isclose(
                sum(float(value) for value in probabilities.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("solution probabilities must sum to one")
            if float(self.success_probability) <= 0.0:
                raise ValueError(
                    "a reported solution requires a positive success probability"
                )
        elif float(self.success_probability) > 0.0:
            raise ValueError(
                "solution probabilities must be reported when the ancilla succeeds"
            )

        object.__setattr__(
            self,
            "success_probability",
            float(self.success_probability),
        )
        object.__setattr__(self, "spectral_bound", float(self.spectral_bound))
        object.__setattr__(self, "scaling", float(self.scaling))
        object.__setattr__(
            self,
            "solution_probabilities",
            MappingProxyType(
                {index: float(probabilities[index]) for index in sorted(probabilities)}
            ),
        )

    @property
    def num_unitary_applications(self) -> int:
        """Return the controlled-unitary powers across both QPE blocks.

        The forward and adjoint phase estimations each apply every integer
        power ``U**(2**i)`` for ``i = 0 .. m-1``, so the pair costs
        ``2 * (2**m - 1)`` unitary applications in total. This is the black-box
        query count of the HHL complexity bound and is fixed by
        ``num_evaluation_qubits``.
        """
        return 2 * ((1 << self.num_evaluation_qubits) - 1)


def _assemble(
    problem: LinearSystemProblem,
    *,
    num_evaluation_qubits: int,
    scaling: float,
    approximation_degree: int,
    bound: float | None,
    components: HHLComponents | None,
) -> tuple[Circuit, float]:
    """Assemble the HHL circuit and return it with the spectral bound."""
    if not isinstance(problem, LinearSystemProblem):
        raise TypeError("problem must be a LinearSystemProblem")
    problem.validate()
    normalize_evaluation_qubits(num_evaluation_qubits)
    _validate_scaling(scaling, num_evaluation_qubits=num_evaluation_qubits)
    _validate_approximation_degree(
        approximation_degree,
        num_evaluation_qubits=num_evaluation_qubits,
    )
    if components is None:
        components = HHLComponents()
    else:
        _validate_components(components)

    alpha = _spectral_bound(
        problem.operator,
        bound,
        num_evaluation_qubits=num_evaluation_qubits,
    )
    unitary = _hermitian_unitary(problem.operator, alpha)
    if isinstance(problem.rhs, StatePreparation):
        state_preparation = problem.rhs_circuit
        if state_preparation is None:
            raise ValueError("rhs preparation snapshot is missing")
    else:
        # The registered class accepts a vector; the base ABC leaves init unspecified.
        preparation_constructor = cast(
            Callable[[Sequence[complex]], StatePreparation],
            components.state_preparation,
        )
        state_preparation = snapshot_phase_preparation(
            preparation_constructor(problem.rhs), "rhs"
        )
        _validate_rhs_width(state_preparation, num_qubits=problem.num_qubits)
    inversion = components.eigenvalue_inversion(
        num_evaluation_qubits, scaling
    ).build_circuit()
    forward = components.phase_estimation(
        unitary,
        num_evaluation_qubits,
        approximation_degree=approximation_degree,
        inverse=False,
    ).build_circuit()
    adjoint = components.phase_estimation(
        unitary,
        num_evaluation_qubits,
        approximation_degree=approximation_degree,
        inverse=True,
    ).build_circuit()

    system_width = problem.num_qubits
    block_width = num_evaluation_qubits + system_width
    circuit = Circuit(block_width + 1)
    evaluation_targets = list(circuit.qubits[:num_evaluation_qubits])
    system_targets = list(circuit.qubits[num_evaluation_qubits:block_width])
    ancilla = circuit.qubits[block_width]
    circuit.compose(state_preparation, system_targets)
    circuit.compose(forward, evaluation_targets + system_targets)
    circuit.compose(inversion, evaluation_targets + [ancilla])
    circuit.compose(adjoint, evaluation_targets + system_targets)
    return circuit, alpha


def _reduce_counts(
    request: HHLRequest,
    alpha: float,
    counts: Mapping[str, int],
    shots: int,
) -> HHLResult:
    """Reduce sampled counts to the post-selected solution distribution."""
    system_width = request.problem.num_qubits
    low_mask = (1 << system_width) - 1
    success_count = 0
    observed: dict[int, int] = {}
    for bitstring, count in counts.items():
        if count <= 0:
            continue
        outcome = int(bitstring, 2)
        if (outcome >> system_width) & 1:
            success_count += count
            index = outcome & low_mask
            observed[index] = observed.get(index, 0) + count

    probabilities = (
        {index: value / success_count for index, value in observed.items()}
        if success_count
        else {}
    )
    return HHLResult(
        success_probability=success_count / shots,
        solution_probabilities=probabilities,
        spectral_bound=alpha,
        scaling=float(request.scaling),
        num_evaluation_qubits=request.num_evaluation_qubits,
        num_system_qubits=system_width,
        shots=shots,
    )


@final
class HHL:
    """Solve ``A x = b`` with the classic phase-estimation HHL algorithm.

    The operator must be Hermitian positive definite. ``|b>`` is prepared in
    the system register, phase estimation of
    ``U = exp(2*pi*i*A/alpha)`` writes the binary fraction of each
    ``lambda_j / alpha`` into the evaluation register, a controlled reciprocal
    rotation entangles the ancilla with the eigenvalue, and the inverse phase
    estimation uncomputes the evaluation register. Post-selecting the ancilla
    on ``|1>`` leaves the system register in a state proportional to
    ``A**-1 |b>`` [1].

    The default spectral bound shrinks the largest eigenvalue by one
    phase-grid step, so that eigenvalue stays resolvable instead of rounding
    onto the phase-zero grid point, where the reciprocal encoding would drop
    it. A ``bound`` that pushes any eigenvalue closer to the period endpoint
    than half a grid step triggers a ``UserWarning``. The reciprocal amplitude
    scaling bounds the post-selection success probability, which is
    ``O(1/kappa**2)`` up to the scaling factor.

    Registers are little-endian. ``circuit.qubits[0:m]`` is the evaluation
    register, ``circuit.qubits[m:m+n]`` is the system register, and
    ``circuit.qubits[m+n]`` is the ancilla; the total width is ``m + n + 1``.
    The batch is validated before one flattened ``Sampler`` call, and each
    reported solution is the system distribution over the shots whose ancilla
    outcome is ``1``.

    Args:
        sampler: Sampling implementation used to execute the circuits.
        components: Operator class registry, or ``None`` for the defaults.

    Raises:
        TypeError: If ``sampler``, ``components``, or a request has the wrong
            type.
        ValueError: If a request invariant or sampler-result contract fails.

    Warns:
        UserWarning: If a request resolves a bound that leaves an eigenvalue
            within one phase-grid step of the period endpoint.
    """

    __slots__ = ("_components", "_sampler")

    def __init__(
        self,
        sampler: Sampler,
        *,
        components: HHLComponents | None = None,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler instance")
        if components is None:
            components = HHLComponents()
        else:
            _validate_components(components)
        self._sampler = sampler
        self._components = components

    @property
    def sampler(self) -> Sampler:
        """Return the injected Sampler implementation."""
        return self._sampler

    @property
    def components(self) -> HHLComponents:
        """Return the registered operator classes."""
        return self._components

    @staticmethod
    def build_circuit(
        problem: LinearSystemProblem,
        *,
        num_evaluation_qubits: int,
        scaling: float,
        approximation_degree: int = 0,
        bound: float | None = None,
        components: HHLComponents | None = None,
    ) -> Circuit:
        """Build a fresh HHL circuit from a linear-system problem and operator classes.

        Args:
            problem: LinearSystemProblem containing the operator and a right-hand
                vector or fixed StatePreparation.
            num_evaluation_qubits: Positive phase-register width.
            scaling: Reciprocal rotation scale in ``(0, 2**-num_evaluation_qubits]``.
            approximation_degree: Inverse-QFT approximation used in both phase blocks.
            bound: Optional positive spectral bound for eigenvalue encoding.
            components: Registered operator classes, or None for the defaults.

        Returns:
            An unmeasured circuit with evaluation, system, and reciprocal-flag
            registers in that order. Sampling and postselection are separate.

        Raises:
            TypeError: If the problem, settings, or registered classes have wrong types.
            ValueError: If the system, bindings, spectral scaling, or widths are invalid.
        """
        circuit, _ = _assemble(
            problem,
            num_evaluation_qubits=num_evaluation_qubits,
            scaling=scaling,
            approximation_degree=approximation_degree,
            bound=bound,
            components=components,
        )
        return circuit

    @overload
    def run(
        self,
        problem: LinearSystemProblem,
        *,
        num_evaluation_qubits: int,
        scaling: float,
        shots: int,
        approximation_degree: int = 0,
        bound: float | None = None,
        seed: int | None = None,
    ) -> HHLResult: ...

    @overload
    def run(self, requests: Sequence[HHLRequest], /) -> list[HHLResult]: ...

    def run(
        self,
        problem: LinearSystemProblem | Sequence[HHLRequest],
        *,
        num_evaluation_qubits: int | None = None,
        scaling: float | None = None,
        shots: int | None = None,
        approximation_degree: int | None = None,
        bound: float | None = None,
        seed: int | None = None,
    ) -> HHLResult | list[HHLResult]:
        """Sample one post-selected solution distribution or an ordered batch.

        Args:
            problem: Linear system or positional batch with per-request settings.
            num_evaluation_qubits: Phase-register width for one system.
            scaling: Reciprocal rotation scale in (0, 2**-num_evaluation_qubits].
            shots: Total samples before post-selection.
            approximation_degree: Small-angle QFT layers to omit in both phase blocks.
            bound: Optional spectral bound used to encode eigenvalues.
            seed: Optional sampler seed for the single system.

        Returns:
            Success probability and conditional basis probabilities, or a list in batch order.

        Raises:
            TypeError: If inputs or single/batch options are incompatible.
            ValueError: If the system, spectral settings or sampler result is invalid.
        """
        if isinstance(problem, LinearSystemProblem):
            if num_evaluation_qubits is None or scaling is None or shots is None:
                raise TypeError(
                    "a single problem requires num_evaluation_qubits, scaling and shots"
                )
            requests: Sequence[HHLRequest] = (
                HHLRequest(
                    problem,
                    num_evaluation_qubits,
                    scaling,
                    shots,
                    approximation_degree=0
                    if approximation_degree is None
                    else approximation_degree,
                    bound=bound,
                    seed=seed,
                ),
            )
        else:
            if any(
                value is not None
                for value in (
                    num_evaluation_qubits,
                    scaling,
                    shots,
                    approximation_degree,
                    bound,
                    seed,
                )
            ):
                raise TypeError("batch requests specify their own HHL settings")
            requests = problem

        assembled: list[tuple[HHLRequest, Circuit, float]] = []
        for request in requests:
            if not isinstance(request, HHLRequest):
                raise TypeError("requests must contain HHLRequest instances")
            request.validate()
            circuit, alpha = _assemble(
                request.problem,
                num_evaluation_qubits=request.num_evaluation_qubits,
                scaling=request.scaling,
                approximation_degree=request.approximation_degree,
                bound=request.bound,
                components=self._components,
            )
            assembled.append((request, circuit, alpha))
        if not assembled:
            return []

        sampler_requests: list[SamplerRequest] = []
        for request, circuit, _ in assembled:
            block_width = request.num_evaluation_qubits + request.problem.num_qubits
            measured = tuple(
                circuit.qubits[request.num_evaluation_qubits : block_width + 1]
            )
            sampler_requests.append(
                SamplerRequest(circuit, measured, request.shots, request.seed)
            )

        sampled = require_sampler_results(
            self._sampler.run(sampler_requests),
            sampler_requests,
        )

        results: list[HHLResult] = []
        for (request, _circuit, alpha), sample in zip(assembled, sampled):
            results.append(_reduce_counts(request, alpha, sample.counts, sample.shots))
        return results[0] if isinstance(problem, LinearSystemProblem) else results
