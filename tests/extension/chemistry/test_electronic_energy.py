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

"""Independent Fock-space tests for electronic-energy conversion."""

import numpy as np
import pytest
from cqlib.qis import (
    FermionicOp,
    Hamiltonian,
    JordanWignerMapper,
    PauliString,
    Phase,
)

from cqlib_algorithms.extension.chemistry import ElectronicEnergy, ElectronicIntegrals


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    return (state | (1 << mode) if create else state & ~(1 << mode), sign)


def _apply_product(
    state: int,
    operations: tuple[tuple[bool, int], ...],
) -> tuple[int, float] | None:
    amplitude = 1.0
    output = state
    for create, mode in reversed(operations):
        applied = _apply_ladder(output, mode, create=create)
        if applied is None:
            return None
        output, sign = applied
        amplitude *= sign
    return output, amplitude


def _independent_fock_matrix(integrals: ElectronicIntegrals) -> np.ndarray:
    n = integrals.num_spatial_orbitals
    size = 1 << (2 * n)
    matrix = np.eye(size, dtype=np.complex128) * integrals.constant

    for basis in range(size):
        for p in range(n):
            for q in range(n):
                for sigma in range(2):
                    applied = _apply_product(
                        basis, ((True, 2 * p + sigma), (False, 2 * q + sigma))
                    )
                    if applied is not None:
                        output, sign = applied
                        matrix[output, basis] += integrals.one_body[p, q] * sign
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        coefficient = 0.5 * integrals.two_body[p, q, r, s]
                        for sigma in range(2):
                            for tau in range(2):
                                applied = _apply_product(
                                    basis,
                                    (
                                        (True, 2 * p + sigma),
                                        (True, 2 * r + tau),
                                        (False, 2 * s + tau),
                                        (False, 2 * q + sigma),
                                    ),
                                )
                                if applied is not None:
                                    output, sign = applied
                                    matrix[output, basis] += coefficient * sign
    return matrix


def _symmetric_eri(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n,) * 4)
    return (
        sum(
            (
                raw,
                raw.transpose(1, 0, 2, 3),
                raw.transpose(0, 1, 3, 2),
                raw.transpose(1, 0, 3, 2),
                raw.transpose(2, 3, 0, 1),
                raw.transpose(3, 2, 0, 1),
                raw.transpose(2, 3, 1, 0),
                raw.transpose(3, 2, 1, 0),
            )
        )
        / 8.0
    )


@pytest.mark.reference
def test_one_spatial_orbital_has_analytic_spectrum():
    constant = 0.4
    h = -1.2
    interaction = 0.7
    energy = ElectronicEnergy(ElectronicIntegrals(constant, [[h]], [[[[interaction]]]]))

    matrix = energy.to_qubit_hamiltonian().to_matrix()

    assert np.allclose(
        matrix,
        np.diag([constant, constant + h, constant + h, constant + 2 * h + interaction]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.reference
def test_two_orbital_core_mapping_matches_independent_fock_construction():
    rng = np.random.default_rng(44)
    raw_one = rng.normal(size=(2, 2))
    one_body = (raw_one + raw_one.T) / 2.0
    integrals = ElectronicIntegrals(0.19, one_body, _symmetric_eri(45, 2))
    energy = ElectronicEnergy(integrals)

    expected = _independent_fock_matrix(integrals)
    actual = energy.to_qubit_hamiltonian().to_matrix()

    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(actual, actual.conj().T, rtol=0.0, atol=1e-12)
    assert energy.to_fermionic_op().is_hermitian(atol=1e-12)


def test_mapper_is_optional_and_must_follow_core_protocol():
    energy = ElectronicEnergy(ElectronicIntegrals(0.0, [[1.0]], np.zeros((1,) * 4)))
    expected = JordanWignerMapper().map_operator(energy.to_fermionic_op())
    assert energy.to_qubit_hamiltonian().equiv(expected, atol=0.0)

    class Mapper:
        def map_operator(self, op: FermionicOp):
            return JordanWignerMapper().map_operator(op)

        def num_qubits(self, num_modes: int) -> int:
            return num_modes

    assert energy.to_qubit_hamiltonian(Mapper()).equiv(expected, atol=0.0)
    with pytest.raises(TypeError, match="mapper"):
        energy.to_qubit_hamiltonian(object())  # type: ignore[arg-type]


class _FixedMapper:
    def __init__(self, width: object, output: object) -> None:
        self.width = width
        self.output = output

    def map_operator(self, op: FermionicOp) -> Hamiltonian:
        return self.output  # type: ignore[return-value]

    def num_qubits(self, num_modes: int) -> int:
        return self.width  # type: ignore[return-value]


def _mapped_hamiltonian(*coefficients: complex) -> Hamiltonian:
    result = Hamiltonian(2)
    for coefficient in coefficients:
        result.add_term(PauliString.from_str("XI"), coefficient)
    return result


def test_qubit_mapping_projects_only_bounded_aggregated_imaginary_roundoff():
    energy = ElectronicEnergy(ElectronicIntegrals(0.0, [[1.0]], np.zeros((1,) * 4)))
    small_residual = _mapped_hamiltonian(1.5 + 5e-13j)
    projected = energy.to_qubit_hamiltonian(_FixedMapper(2, small_residual))
    np.testing.assert_allclose(
        projected.to_matrix(),
        1.5 * PauliString.from_str("XI").to_matrix(),
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.abs(projected.to_matrix() - small_residual.to_matrix())) < 1e-12

    canceling = _mapped_hamiltonian(1.0 + 1e-4j, 2.0 - 1e-4j)
    canceled = energy.to_qubit_hamiltonian(_FixedMapper(2, canceling))
    np.testing.assert_allclose(
        canceled.to_matrix(),
        3.0 * PauliString.from_str("XI").to_matrix(),
        rtol=0.0,
        atol=0.0,
    )

    phased_pauli = PauliString.from_str("XI")
    phased_pauli.phase = Phase.i()
    phased = Hamiltonian(2)
    phased.add_term(phased_pauli, -2.0j)
    absorbed = energy.to_qubit_hamiltonian(_FixedMapper(2, phased))
    np.testing.assert_allclose(
        absorbed.to_matrix(),
        2.0 * PauliString.from_str("XI").to_matrix(),
        rtol=0.0,
        atol=0.0,
    )
    output_pauli, output_coefficient = absorbed.terms[0]
    assert output_pauli.phase == Phase.plus()
    assert output_coefficient == 2.0


def test_qubit_mapping_rejects_nonhermitian_nonfinite_and_invalid_outputs():
    energy = ElectronicEnergy(ElectronicIntegrals(0.0, [[1.0]], np.zeros((1,) * 4)))
    with pytest.raises(ValueError, match="not Hermitian"):
        energy.to_qubit_hamiltonian(_FixedMapper(2, _mapped_hamiltonian(1.0 + 2e-12j)))
    with pytest.raises(ValueError, match="finite"):
        energy.to_qubit_hamiltonian(
            _FixedMapper(2, _mapped_hamiltonian(complex(np.nan)))
        )
    with pytest.raises(TypeError, match="return a Hamiltonian"):
        energy.to_qubit_hamiltonian(_FixedMapper(2, object()))
    with pytest.raises(ValueError, match="width"):
        energy.to_qubit_hamiltonian(_FixedMapper(1, _mapped_hamiltonian(1.0)))
    with pytest.raises(TypeError, match="return an int"):
        energy.to_qubit_hamiltonian(_FixedMapper("2", _mapped_hamiltonian(1.0)))
    with pytest.raises(ValueError, match="positive width"):
        energy.to_qubit_hamiltonian(_FixedMapper(0, Hamiltonian(1)))


def test_qubit_mapping_result_is_independent_of_mapper_output_mutation():
    energy = ElectronicEnergy(ElectronicIntegrals(0.0, [[1.0]], np.zeros((1,) * 4)))
    source = _mapped_hamiltonian(1.0)
    mapped = energy.to_qubit_hamiltonian(_FixedMapper(2, source))
    source.add_term(PauliString.from_str("ZI"), 4.0)
    assert mapped.num_terms == 1
