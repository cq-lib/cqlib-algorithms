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

r"""Quantum subspace expansion error mitigation.

Subspace expansion starts from the state :math:`|\psi\rangle` prepared by the
noisy circuit and enlarges it to the subspace spanned by
:math:`\{B_j |\psi\rangle\}`, where each :math:`B_j` is a product of a subset
of the supplied check operators. Projecting onto that subspace gives the
overlap matrix :math:`S_{jk} = \langle\psi| B_j^\dagger B_k |\psi\rangle` and
the projected Hamiltonian
:math:`H_{jk} = \langle\psi| B_j^\dagger H_\text{code} B_k |\psi\rangle`,
which define the generalized eigenvalue problem :math:`H v = E S v`. Its
lowest eigenvalue is the corrected energy, and the same ground vector projects
any target observable onto the subspace as

.. math::

    \langle O \rangle_S = \frac{v^\dagger O_S v}{v^\dagger S v},
    \qquad
    (O_S)_{jk} = \langle\psi| B_j^\dagger O B_k |\psi\rangle .

which equals the corrected energy when the observable is the code
Hamiltonian.

Every entry is an expectation value of a Pauli product in the fixed state
:math:`|\psi\rangle`, so the check operators are absorbed into the measured
Pauli string rather than appended to the circuit. The effective Hamiltonian
and observable operators are read on the exact estimator surface; the
code-space overlaps are read on the counting sampler surface and recombined
with ``PauliString.expectation_probs``. When the sampled overlap loses
positive definiteness it is shifted symmetrically and a
:class:`~cqlib_algorithms.error_mitigation.inference.ConvergenceWarning` is
emitted, because a non-positive overlap signals sampling noise larger than the
subspace signal.

:func:`generalized_eigenvalue` solves the generalized problem with NumPy
alone, :func:`execute_with_qse` runs the mitigation once, and
:func:`qse_executor` exposes the same computation as a reusable executor so
the technique can be chained on top of another one.

References
----------
[1] J. R. McClean, M. E. Kimchi-Schwartz, J. Carter, and W. A. de Jong, "Hybrid
    quantum-classical hierarchy for mitigation of decoherence and determination of
    excited states," Physical Review A 95, 042308, 2017. arXiv:1603.05681.
[2] J. R. McClean, Z. Jiang, N. C. Rubin, R. Babbush, and H. Neven, "Decoding quantum
    errors with subspace expansions," Nature Communications 11, 636, 2020.
    arXiv:1903.05786.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np

from cqlib import Circuit
from cqlib.circuit import StandardGate, ValueOperation
from cqlib.qis import Hamiltonian, PauliString, Phase

from ...protocols import Estimator, EstimatorRequest, EstimatorResult, SamplerRequest
from .._circuits import rebuild, strip_measurements
from ..core import QemExecutor
from ..inference import ConvergenceWarning

# Symmetric shift applied to the overlap matrix when its smallest eigenvalue is
# not positive. A positive overlap is required by the Cholesky whitening below;
# the shift both restores definiteness and records how far the sampled matrix
# was from a physical subspace metric.
_DEFAULT_REGULARIZATION = 1.0e-8


def generalized_eigenvalue(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    /,
    *,
    regularization: float = 1.0e-8,
) -> float:
    r"""Return the smallest eigenvalue of :math:`H v = E S v`.

    The problem is reduced to a standard symmetric one by Cholesky whitening:
    with :math:`S = L L^\dagger`, the substitution :math:`v = L^{-\dagger} y`
    turns :math:`H v = E S v` into
    :math:`L^{-1} H L^{-\dagger} y = E y`, whose spectrum is computed with
    :meth:`numpy.linalg.eigh`. Because :math:`S` must be positive definite, a
    symmetric shift is applied first whenever its smallest eigenvalue is not
    positive, and a warning is emitted.

    Args:
        hamiltonian: Square Hermitian matrix :math:`H` of shape ``(d, d)``.
        overlap: Square Hermitian matrix :math:`S` of shape ``(d, d)``, expected to be
            positive definite.
        regularization: Positive shift that raises a non-positive :math:`S` above zero before
            whitening. Ignored when :math:`S` is already positive definite.

    Returns:
        The smallest generalized eigenvalue, the corrected energy.

    Raises:
        TypeError: If ``hamiltonian`` or ``overlap`` is not a NumPy array, or if
            ``regularization`` is not a real number.
        ValueError: If a matrix is not square, the two shapes differ, or
            ``regularization`` is not finite or non-positive.

    Warns:
        ConvergenceWarning: If the smallest eigenvalue of ``overlap`` is not positive, in which
            case a symmetric shift of ``regularization`` is applied.

    """
    matrix_h = _as_square_matrix(hamiltonian, "hamiltonian")
    matrix_s = _as_square_matrix(overlap, "overlap")
    if matrix_h.shape != matrix_s.shape:
        raise ValueError("hamiltonian and overlap must have the same shape")
    _validate_regularization(regularization)

    value, _, _ = _smallest_generalized_eigenpair(
        matrix_h,
        matrix_s,
        regularization,
    )
    return value


def execute_with_qse(
    circuit: Circuit,
    executor: QemExecutor,
    observable: Hamiltonian | PauliString,
    /,
    *,
    check_operators: Sequence[PauliString],
    code_hamiltonian: Hamiltonian,
    shots: int | None = None,
    seed: int | None = None,
) -> float:
    r"""Estimate one observable on ``circuit`` with subspace expansion.

    This is the immediate form of :func:`qse_executor`::

        execute_with_qse(c, ex, obs, check_operators=ops,
                         code_hamiltonian=h)
        == qse_executor(ex, check_operators=ops, code_hamiltonian=h).estimate(
               [EstimatorRequest(c, obs)]
           )[0].expectation

    The returned value is the projected expectation
    :math:`(v^\dagger O_S v)/(v^\dagger S v)` of ``observable`` in the ground
    vector of the generalized problem built from ``code_hamiltonian``.

    Args:
        circuit: Fully bound circuit preparing the noisy state :math:`|\psi\rangle`.
            Any trailing measurements are stripped before evaluation.
        executor: Execution backend supplying the estimator and sampler surfaces.
        observable: Observable to estimate, a Hermitian Hamiltonian or Pauli string acting
            on exactly the circuit's qubits.
        check_operators: Non-empty Pauli strings whose subset products span the subspace. They
            must act on exactly the circuit's qubits.
        code_hamiltonian: Hermitian Hamiltonian defining the projected energy functional, acting
            on exactly the circuit's qubits.
        shots: Shot count for the sampled overlap matrix, or ``None`` to evaluate the
            overlaps exactly on the estimator surface.
        seed: Seed of the sampled overlap matrix, or ``None`` for fresh randomness.
            Every Pauli overlap in one run derives its draw from this value, so the
            run is reproducible.

    Returns:
        The mitigated expectation value of ``observable``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``check_operators`` is empty, if ``code_hamiltonian`` or a check
            operator acts on a different width than the circuit, or if ``shots`` is
            not positive.

    Warns:
        ConvergenceWarning: If the sampled overlap matrix is not positive definite, in which case
            it is shifted before the generalized problem is solved.

    """
    wrapped = qse_executor(
        executor,
        check_operators=check_operators,
        code_hamiltonian=code_hamiltonian,
        shots=shots,
        seed=seed,
    )
    request = EstimatorRequest(circuit=circuit, observable=observable)
    return wrapped.estimate([request])[0].expectation


def qse_executor(
    executor: QemExecutor,
    /,
    *,
    check_operators: Sequence[PauliString],
    code_hamiltonian: Hamiltonian,
    shots: int | None = None,
    seed: int | None = None,
) -> QemExecutor:
    r"""Return an executor with subspace expansion applied.

    The returned executor keeps the wrapped executor's sampling surface and
    caching flag, so QSE can be chained on top of another technique. Its
    expectation surface projects the requested observable onto the subspace
    defined by ``check_operators`` and ``code_hamiltonian``; its sampling
    surface is the one supplying the overlap measurements.

    Args:
        executor: Execution backend to wrap.
        check_operators: Non-empty Pauli strings whose subset products span the subspace.
        code_hamiltonian: Hermitian Hamiltonian defining the projected energy functional.
        shots: Shot count for the sampled overlap matrix, or ``None`` to evaluate the
            overlaps exactly on the estimator surface.
        seed: Seed of the sampled overlap matrix, or ``None`` to defer to the
            request's own seed.

    Returns:
        Executor whose expectation surface applies subspace expansion.

    Raises:
        TypeError: If ``check_operators`` is not a sequence of Pauli strings, if
            ``code_hamiltonian`` is not a Hamiltonian, or if ``shots`` or ``seed``
            has the wrong type.
        ValueError: If ``check_operators`` is empty or ``shots`` is not positive.

    """
    _validate_check_operators(check_operators)
    _validate_code_hamiltonian(code_hamiltonian)
    _validate_shots(shots)
    _validate_seed(seed)

    operators = tuple(check_operators)
    estimator = _QseEstimator(
        executor,
        check_operators=operators,
        code_hamiltonian=code_hamiltonian,
        shots=shots,
        seed=seed,
    )
    return QemExecutor(
        estimator=estimator,
        sampler=executor.sampler,
        use_cache=executor.use_cache,
    )


class _QseEstimator(Estimator):
    r"""Expectation surface that projects onto a check-operator subspace.

    Args:
        executor: Execution backend supplying both measurement surfaces.
        check_operators: Non-empty Pauli strings defining the subspace basis.
        code_hamiltonian: Hermitian Hamiltonian defining the projected energy functional.
        shots: Shot count for the sampled overlap matrix, or ``None`` for an exact
            overlap on the estimator surface.
        seed: Seed of the sampled overlap matrix, or ``None`` to defer to the
            request's own seed.

    """

    __slots__ = (
        "_basis",
        "_check_operators",
        "_code_hamiltonian",
        "_executor",
        "_seed",
        "_shots",
    )

    _basis: list[PauliString]
    _check_operators: tuple[PauliString, ...]
    _code_hamiltonian: Hamiltonian
    _executor: QemExecutor
    _seed: int | None
    _shots: int | None

    def __init__(
        self,
        executor: QemExecutor,
        /,
        *,
        check_operators: tuple[PauliString, ...],
        code_hamiltonian: Hamiltonian,
        shots: int | None,
        seed: int | None,
    ) -> None:
        self._executor = executor
        self._check_operators = check_operators
        self._code_hamiltonian = code_hamiltonian
        self._shots = shots
        self._seed = seed
        self._basis = _subspace_basis(check_operators)

    def run(
        self,
        requests: Sequence[EstimatorRequest],
        /,
    ) -> list[EstimatorResult]:
        r"""Mitigate every request and return results in input order."""
        return [self._mitigate(request) for request in requests]

    def _mitigate(self, request: EstimatorRequest) -> EstimatorResult:
        r"""Build the projected matrices and return the mitigated estimate."""
        circuit, _ = strip_measurements(request.circuit)
        observable = _as_hamiltonian(request.observable)
        _validate_widths(circuit, observable, self._code_hamiltonian, self._check_operators)
        seed = self._seed if self._seed is not None else request.seed

        dimension = len(self._basis)
        overlap = np.zeros((dimension, dimension), dtype=np.complex128)
        hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
        projected = np.zeros((dimension, dimension), dtype=np.complex128)

        # The same Pauli expectation recurs across many matrix entries, so a
        # per-request cache keeps the number of executions to the distinct
        # Pauli strings seen in the state |psi>.
        cache: dict[tuple[int, int, bool], float] = {}
        for row in range(dimension):
            for column in range(dimension):
                left = self._basis[row]
                right = self._basis[column]
                overlap[row, column] = self._overlap_entry(circuit, left, right, cache, seed)
                hamiltonian[row, column] = self._operator_entry(
                    circuit, left, self._code_hamiltonian, right, cache, seed
                )
                projected[row, column] = self._operator_entry(
                    circuit, left, observable, right, cache, seed
                )

        _, vector, effective_overlap = _smallest_generalized_eigenpair(
            hamiltonian,
            overlap,
            _DEFAULT_REGULARIZATION,
        )
        numerator = complex(vector.conj() @ projected @ vector).real
        denominator = complex(vector.conj() @ effective_overlap @ vector).real
        return EstimatorResult(expectation=numerator / denominator)

    def _overlap_entry(
        self,
        circuit: Circuit,
        left: PauliString,
        right: PauliString,
        cache: dict[tuple[int, int, bool], float],
        seed: int | None,
    ) -> complex:
        r"""Return :math:`\langle\psi| B_\text{left}^\dagger B_\text{right}
        |\psi\rangle`.

        The subset product is a single Pauli string with a global phase, so its
        expectation is that phase times the expectation of the phase-free Pauli.
        """
        product = left * right
        omega, free = _phase_free(product)
        value = self._expectation(circuit, free, sampled=True, cache=cache, seed=seed)
        return omega * value

    def _operator_entry(
        self,
        circuit: Circuit,
        left: PauliString,
        operator: Hamiltonian,
        right: PauliString,
        cache: dict[tuple[int, int, bool], float],
        seed: int | None,
    ) -> complex:
        r"""Return :math:`\langle\psi| B_\text{left}^\dagger O B_\text{right}
        |\psi\rangle` for a Hamiltonian ``operator``.

        Each Pauli term is conjugated by the two check-operator products and
        carries a global phase; the phase-free part is measured and the
        coefficient, phase and Pauli expectation are combined classically.
        """
        total = 0.0j
        for pauli, coefficient in operator.terms:
            product = left * pauli * right
            omega, free = _phase_free(product)
            value = self._expectation(circuit, free, sampled=False, cache=cache, seed=seed)
            total += complex(coefficient) * omega * value
        return total

    def _expectation(
        self,
        circuit: Circuit,
        pauli: PauliString,
        *,
        sampled: bool,
        cache: dict[tuple[int, int, bool], float],
        seed: int | None,
    ) -> float:
        r"""Return :math:`\langle\psi| P |\psi\rangle` for a phase-free Pauli.

        ``sampled`` selects the sampling surface when a shot count is bound and
        the exact estimator surface otherwise.
        """
        key = (pauli.x_mask, pauli.z_mask, sampled)
        if key in cache:
            return cache[key]
        # The sampling surface is reachable only with a bound shot count, so the
        # count is read once here and handed to the request builder explicitly.
        shots = self._shots
        if sampled and shots is not None:
            value = self._sample_expectation(circuit, pauli, seed, shots)
        else:
            value = self._estimate_expectation(circuit, pauli, seed)
        cache[key] = value
        return value

    def _sample_expectation(
        self,
        circuit: Circuit,
        pauli: PauliString,
        seed: int | None,
        shots: int,
    ) -> float:
        r"""Return the Pauli expectation from a rotated sampled distribution."""
        rotated = _rotate_into_basis(circuit, pauli)
        result = self._executor.sample(
            [
                SamplerRequest(
                    circuit=rotated,
                    qubits=tuple(rotated.qubits),
                    shots=shots,
                    seed=_pauli_seed(seed, pauli),
                )
            ]
        )[0]
        distribution = {
            bitstring: count / result.shots
            for bitstring, count in result.counts.items()
        }
        # The circuit was rotated so the Pauli acts diagonally on the measured
        # basis, so the recorded bits are its +1/-1 eigenvalues and the
        # measurement basis passed to the recombination matches the Pauli.
        return float(pauli.expectation_probs([(pauli, distribution)]))

    def _estimate_expectation(
        self, circuit: Circuit, pauli: PauliString, seed: int | None
    ) -> float:
        r"""Return the exact Pauli expectation from the estimator surface."""
        request = EstimatorRequest(
            circuit=circuit, observable=pauli, seed=_pauli_seed(seed, pauli)
        )
        return float(self._executor.estimate([request])[0].expectation)


def _smallest_generalized_eigenpair(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Return the smallest eigenpair of :math:`H v = E S v` and the shifted S.

    The matrices are Hermitised, a symmetric shift makes :math:`S` positive
    definite, and the problem is reduced to a standard one by Cholesky
    whitening :math:`S = L L^\dagger` before :meth:`numpy.linalg.eigh` extracts
    the lowest eigenpair of :math:`L^{-1} H L^{-\dagger}`.

    Returns:
        The smallest eigenvalue, the matching eigenvector in the original
            coordinates, and the overlap matrix actually used, which is the shifted
            one when regularization was triggered.

    """
    symmetric_h = 0.5 * (hamiltonian + hamiltonian.conj().T)
    symmetric_s = 0.5 * (overlap + overlap.conj().T)

    smallest_overlap = float(np.linalg.eigvalsh(symmetric_s)[0])
    if smallest_overlap <= 0.0:
        shift = regularization - smallest_overlap
        symmetric_s = symmetric_s + shift * np.eye(symmetric_s.shape[0])
        warnings.warn(
            "The projected overlap matrix is not positive definite: its "
            f"smallest eigenvalue is {smallest_overlap:.3e} <= 0, which means "
            "the sampling noise is larger than the subspace signal. The "
            f"overlap was shifted by {shift:.3e} before solving, so the "
            "corrected value is unreliable; increase the shot count or shrink "
            "the check-operator set. See McClean 2020, arXiv:1903.05786.",
            ConvergenceWarning,
            stacklevel=2,
        )

    factor = np.linalg.cholesky(symmetric_s)
    whitened = np.linalg.solve(factor, symmetric_h)
    standard = np.linalg.solve(factor, whitened.conj().T).conj().T
    standard = 0.5 * (standard + standard.conj().T)

    values, vectors = np.linalg.eigh(standard)
    value = float(values[0])
    # Invert the whitening substitution v = L^{-dagger} y for the ground y.
    vector = np.linalg.solve(factor.conj().T, vectors[:, 0])
    return value, vector, symmetric_s


def _subspace_basis(check_operators: tuple[PauliString, ...]) -> list[PauliString]:
    r"""Return the phase-free subset products spanning the subspace.

    The empty subset contributes the identity, so the source state is always
    part of the subspace. Products that share the same Pauli content up to a
    global phase describe the same basis vector and are kept once, with the
    phase dropped because it does not affect the subspace.
    """
    num_qubits = check_operators[0].num_qubits
    basis: list[PauliString] = []
    seen: set[tuple[int, int]] = set()
    for mask in range(1 << len(check_operators)):
        product = PauliString(num_qubits)
        for bit, operator in enumerate(check_operators):
            if mask & (1 << bit):
                product = product * operator
        key = (product.x_mask, product.z_mask)
        if key in seen:
            continue
        seen.add(key)
        product.phase = Phase.plus()
        basis.append(product)
    return basis


def _phase_free(pauli: PauliString) -> tuple[complex, PauliString]:
    r"""Split a Pauli string into its global phase and a phase-free copy."""
    omega = complex(pauli.phase.to_complex())
    free = pauli.copy()
    free.phase = Phase.plus()
    return omega, free


def _rotate_into_basis(circuit: Circuit, pauli: PauliString) -> Circuit:
    r"""Extend ``circuit`` with the rotations reading ``pauli`` in the Z basis.

    An :math:`X` factor needs a Hadamard, a :math:`Y` factor needs
    :math:`S^\dagger` followed by a Hadamard, and a :math:`Z` factor is already
    diagonal; identity factors need nothing.
    """
    rotations: list[ValueOperation] = []
    for index, qubit in enumerate(circuit.qubits):
        x_bit = (pauli.x_mask >> index) & 1
        z_bit = (pauli.z_mask >> index) & 1
        if x_bit and not z_bit:
            rotations.append(
                ValueOperation.from_standard_gate(StandardGate.H, [qubit])
            )
        elif x_bit and z_bit:
            rotations.append(
                ValueOperation.from_standard_gate(StandardGate.SDG, [qubit])
            )
            rotations.append(
                ValueOperation.from_standard_gate(StandardGate.H, [qubit])
            )
    return rebuild(circuit, [*circuit.operations, *rotations])


def _pauli_seed(seed: int | None, pauli: PauliString) -> int | None:
    r"""Return the child seed one sampled Pauli draws from.

    Args:
        seed: Run seed, or ``None`` for fresh randomness.
        pauli: Pauli whose expectation is sampled.

    Returns:
        A seed that depends only on ``seed`` and ``pauli``, so distinct
            overlaps draw independently while one run seed still reproduces the
            whole overlap matrix; ``None`` when ``seed`` is ``None``.

    """
    if seed is None:
        return None
    sequence = np.random.SeedSequence([seed, pauli.x_mask, pauli.z_mask])
    return int(sequence.generate_state(1)[0])


def _as_hamiltonian(observable: Hamiltonian | PauliString, /) -> Hamiltonian:
    r"""Wrap a bare Pauli string so every observable shares one code path."""
    if isinstance(observable, PauliString):
        return Hamiltonian.from_pauli(observable)
    return observable


def _as_square_matrix(value: np.ndarray, name: str) -> np.ndarray:
    r"""Return a square matrix as a complex array, or raise on a bad shape."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return np.asarray(value, dtype=np.complex128)


def _validate_regularization(regularization: float) -> None:
    r"""Validate the subspace regularization."""
    if isinstance(regularization, bool) or not isinstance(
        regularization, (int, float, np.floating)
    ):
        raise TypeError("regularization must be a real number")
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be positive and finite")


def _validate_check_operators(check_operators: Sequence[PauliString]) -> None:
    r"""Validate the check-operator sequence."""
    if isinstance(check_operators, (str, bytes)) or not isinstance(
        check_operators, Sequence
    ):
        raise TypeError("check_operators must be a sequence of Pauli strings")
    if len(check_operators) == 0:
        raise ValueError("check_operators must contain at least one Pauli string")
    for operator in check_operators:
        if not isinstance(operator, PauliString):
            raise TypeError("check_operators must contain only cqlib.qis.PauliString")


def _validate_code_hamiltonian(code_hamiltonian: Hamiltonian) -> None:
    r"""Validate a code Hamiltonian."""
    if not isinstance(code_hamiltonian, Hamiltonian):
        raise TypeError("code_hamiltonian must be a cqlib.qis.Hamiltonian")


def _validate_shots(shots: int | None) -> None:
    r"""Validate a positive shot count."""
    if shots is None:
        return
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer or None")
    if shots <= 0:
        raise ValueError("shots must be greater than zero")


def _validate_seed(seed: int | None) -> None:
    r"""Validate an optional random seed."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _validate_widths(
    circuit: Circuit,
    observable: Hamiltonian,
    code_hamiltonian: Hamiltonian,
    check_operators: tuple[PauliString, ...],
) -> None:
    r"""Validate circuit and operator widths."""
    num_qubits = circuit.num_qubits
    if code_hamiltonian.num_qubits != num_qubits:
        raise ValueError(
            "code_hamiltonian cannot act on a different number of qubits than "
            f"the circuit: it acts on {code_hamiltonian.num_qubits} qubit(s) "
            f"but the circuit has {num_qubits}"
        )
    if observable.num_qubits != num_qubits:
        raise ValueError(
            "observable must act on the same number of qubits as the circuit"
        )
    for operator in check_operators:
        if operator.num_qubits != num_qubits:
            raise ValueError(
                "every check operator must act on the same number of qubits as "
                "the circuit"
            )
