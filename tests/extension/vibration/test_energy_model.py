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

"""Independent matrix tests for vibrational-energy conversion."""

from itertools import combinations, product

import numpy as np
import pytest
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_algorithms.extension.many_body import (
    OneHotVibrationalMapper,
    VibrationalOp,
)
from cqlib_algorithms.extension.vibration import (
    ModalIntegrals,
    ModalSpace,
    ModalTransition,
    VibrationalEnergy,
)


def _reference_matrix(integrals: ModalIntegrals, max_order: int) -> np.ndarray:
    space = integrals.modal_space
    matrix = np.eye(space.dimension, dtype=np.complex128) * integrals.constant
    configurations = tuple(
        space.configuration_from_basis_index(index) for index in range(space.dimension)
    )
    row_by_configuration = {
        configuration: row for row, configuration in enumerate(configurations)
    }
    for term, coefficient in integrals.terms:
        if len(term) > max_order:
            continue
        for column, input_configuration in enumerate(configurations):
            if any(
                input_configuration[transition.mode] != transition.ket_modal
                for transition in term
            ):
                continue
            output_configuration = list(input_configuration)
            for transition in term:
                output_configuration[transition.mode] = transition.bra_modal
            row = row_by_configuration[tuple(output_configuration)]
            matrix[row, column] += coefficient
    return matrix


def _random_hermitian_integrals(
    space: ModalSpace,
    *,
    seed: int,
) -> ModalIntegrals:
    rng = np.random.default_rng(seed)
    entries = []
    for order in range(1, space.num_modes + 1):
        for modes in combinations(range(space.num_modes), order):
            dimensions = [space.num_modals[mode] for mode in modes]
            local_dimension = int(np.prod(dimensions))
            raw = rng.normal(
                size=(local_dimension, local_dimension)
            ) + 1.0j * rng.normal(size=(local_dimension, local_dimension))
            block = raw + raw.conjugate().T
            local_configurations = tuple(product(*(range(size) for size in dimensions)))
            for row, bra in enumerate(local_configurations):
                for column, ket in enumerate(local_configurations):
                    coefficient = complex(block[row, column])
                    if coefficient == 0.0:
                        continue
                    entries.append(
                        (
                            tuple(
                                ModalTransition(mode, bra[index], ket[index])
                                for index, mode in enumerate(modes)
                            ),
                            coefficient,
                        )
                    )
    return ModalIntegrals.from_terms(space, entries, constant=-0.31)


class _FixedOneHotMapper(OneHotVibrationalMapper):
    __slots__ = ("_output",)

    def __init__(self, output: Hamiltonian) -> None:
        object.__setattr__(self, "_output", output)

    def map_operator(self, operator: VibrationalOp) -> Hamiltonian:
        del operator
        return self._output


def _mapped_hamiltonian(*coefficients: complex) -> Hamiltonian:
    result = Hamiltonian(2)
    for coefficient in coefficients:
        result.add_term(PauliString.from_str("XI"), coefficient)
    return result


@pytest.mark.reference
@pytest.mark.parametrize(
    ("num_modals", "seed"),
    [((2,), 3), ((2, 2), 5), ((2, 3), 11), ((2, 2, 2), 17)],
)
def test_full_vibrational_conversion_matches_independent_modal_matrix(
    num_modals,
    seed,
):
    space = ModalSpace(num_modals)
    integrals = _random_hermitian_integrals(space, seed=seed)
    energy = VibrationalEnergy(integrals)
    operator = energy.to_vibrational_op()
    hamiltonian = energy.to_qubit_hamiltonian()
    isometry = space.one_hot_isometry()
    expected = _reference_matrix(integrals, space.num_modes)
    mapped = hamiltonian.to_matrix()

    assert operator.preserves_one_hot_subspace()
    assert operator.is_hermitian(atol=1e-12)
    assert hamiltonian.is_hermitian(atol=1e-12)
    np.testing.assert_allclose(operator.to_modal_matrix(), expected, atol=1e-12)
    np.testing.assert_allclose(
        isometry.conjugate().T @ mapped @ isometry,
        expected,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mapped @ isometry,
        isometry @ expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_truncation_excludes_only_couplings_above_selected_order():
    space = ModalSpace((2, 2, 2))
    integrals = _random_hermitian_integrals(space, seed=23)
    energy = VibrationalEnergy(integrals, truncation_order=2)

    assert energy.effective_order == 2
    np.testing.assert_allclose(
        energy.to_vibrational_op().to_modal_matrix(),
        _reference_matrix(integrals, 2),
        rtol=0.0,
        atol=1e-12,
    )


def test_mapper_injection_matches_default_mapper():
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2,)),
        (((ModalTransition(0, 1, 1),), 2.0),),
    )
    energy = VibrationalEnergy(integrals)

    np.testing.assert_array_equal(
        energy.to_qubit_hamiltonian().to_matrix(),
        energy.to_qubit_hamiltonian(OneHotVibrationalMapper()).to_matrix(),
    )


def test_bounded_integral_residual_reaches_an_exactly_hermitian_hamiltonian():
    forward = (ModalTransition(0, 1, 0),)
    backward = (ModalTransition(0, 0, 1),)
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2,)),
        (
            (forward, 0.75 + 0.2j),
            (backward, 0.75 - 0.2j + 4e-13j),
        ),
    )

    hamiltonian = VibrationalEnergy(integrals).to_qubit_hamiltonian()

    assert hamiltonian.is_hermitian(atol=0.0)
    assert all(
        (complex(coefficient) * pauli.phase.to_complex()).imag == 0.0
        for pauli, coefficient in hamiltonian.terms
    )


def test_mapping_projects_only_bounded_aggregated_imaginary_roundoff():
    energy = VibrationalEnergy(
        ModalIntegrals.from_terms(
            ModalSpace((2,)),
            (((ModalTransition(0, 0, 0),), 1.0),),
        )
    )
    small_residual = _mapped_hamiltonian(1.5 + 5e-13j)
    projected = energy.to_qubit_hamiltonian(_FixedOneHotMapper(small_residual))
    np.testing.assert_allclose(
        projected.to_matrix(),
        1.5 * PauliString.from_str("XI").to_matrix(),
        rtol=0.0,
        atol=0.0,
    )
    assert projected.is_hermitian(atol=0.0)

    canceling = _mapped_hamiltonian(1.0 + 1e-4j, 2.0 - 1e-4j)
    canceled = energy.to_qubit_hamiltonian(_FixedOneHotMapper(canceling))
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
    absorbed = energy.to_qubit_hamiltonian(_FixedOneHotMapper(phased))
    output_pauli, output_coefficient = absorbed.terms[0]
    assert output_pauli.phase == Phase.plus()
    assert output_coefficient == 2.0


def test_mapping_rejects_unbounded_nonhermitian_residual():
    energy = VibrationalEnergy(
        ModalIntegrals.from_terms(
            ModalSpace((2,)),
            (((ModalTransition(0, 0, 0),), 1.0),),
        )
    )

    with pytest.raises(ValueError, match="not Hermitian"):
        energy.to_qubit_hamiltonian(
            _FixedOneHotMapper(_mapped_hamiltonian(1.0 + 2e-12j))
        )


def test_invalid_integrals_type_is_rejected():
    with pytest.raises(TypeError, match="integrals must be"):
        VibrationalEnergy(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "error"), [(True, TypeError), (0, ValueError), (3, ValueError)]
)
def test_invalid_truncation_order_is_rejected(value, error):
    integrals = ModalIntegrals.from_terms(
        ModalSpace((2, 2)),
        (((ModalTransition(0, 0, 0),), 1.0),),
    )
    with pytest.raises(error):
        VibrationalEnergy(integrals, value)


def test_invalid_mapper_is_rejected():
    energy = VibrationalEnergy(
        ModalIntegrals.from_terms(
            ModalSpace((1,)),
            (((ModalTransition(0, 0, 0),), 1.0),),
        )
    )
    with pytest.raises(TypeError, match="mapper must be"):
        energy.to_qubit_hamiltonian(object())  # type: ignore[arg-type]
