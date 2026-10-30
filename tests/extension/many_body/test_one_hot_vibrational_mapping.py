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

"""Independent matrix tests for the one-hot vibrational mapping."""

from collections.abc import Sequence

import numpy as np
import pytest

from cqlib_algorithms.extension.many_body import (
    OneHotVibrationalMapper,
    VibrationalLadder,
    VibrationalOp,
)
from cqlib_algorithms.extension.vibration import ModalSpace

I = np.eye(2, dtype=np.complex128)
CREATION = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)


def _configuration(space: ModalSpace, index: int) -> tuple[int, ...]:
    values: list[int] = []
    remainder = index
    for count in space.num_modals:
        values.append(remainder % count)
        remainder //= count
    return tuple(values)


def _operator_from_modal_matrix(
    space: ModalSpace,
    matrix: np.ndarray,
) -> VibrationalOp:
    entries: list[tuple[tuple[VibrationalLadder, ...], complex]] = []
    for column in range(space.dimension):
        input_configuration = _configuration(space, column)
        for row in range(space.dimension):
            coefficient = complex(matrix[row, column])
            if coefficient == 0.0:
                continue
            output_configuration = _configuration(space, row)
            factors: list[VibrationalLadder] = []
            for mode, (output_modal, input_modal) in enumerate(
                zip(output_configuration, input_configuration)
            ):
                factors.extend(
                    (
                        VibrationalLadder("+", mode, output_modal),
                        VibrationalLadder("-", mode, input_modal),
                    )
                )
            entries.append((tuple(factors), coefficient))
    return VibrationalOp.from_terms(space, entries)


def test_single_ladder_uses_qubit_raising_and_lowering_convention():
    space = ModalSpace((2,))
    creation = VibrationalOp.from_terms(
        space,
        (((VibrationalLadder("+", 0, 1),), 1.0),),
    )
    mapped_creation = OneHotVibrationalMapper().map_operator(creation).to_matrix()
    mapped_annihilation = (
        OneHotVibrationalMapper().map_operator(creation.adjoint()).to_matrix()
    )

    np.testing.assert_allclose(mapped_creation, np.kron(CREATION, I), atol=1e-14)
    np.testing.assert_allclose(
        mapped_annihilation,
        np.kron(CREATION.conjugate().T, I),
        atol=1e-14,
    )


def test_one_modal_transition_projects_to_the_expected_matrix_unit():
    space = ModalSpace((3,))
    operator = _operator_from_modal_matrix(
        space,
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 - 3.0j], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        ),
    )
    isometry = space.one_hot_isometry()
    mapped = OneHotVibrationalMapper().map_operator(operator).to_matrix()

    expected = operator.to_modal_matrix()
    np.testing.assert_allclose(
        isometry.conjugate().T @ mapped @ isometry,
        expected,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.reference
@pytest.mark.parametrize("num_modals", [(2,), (3,), (2, 2), (2, 3)])
@pytest.mark.parametrize("seed", [7, 19, 41])
def test_random_hermitian_modal_operator_satisfies_exact_projection_identity(
    num_modals: Sequence[int],
    seed: int,
):
    space = ModalSpace(tuple(num_modals))
    rng = np.random.default_rng(seed + sum(num_modals) * 101 + space.dimension)
    raw = rng.normal(size=(space.dimension, space.dimension)) + 1.0j * rng.normal(
        size=(space.dimension, space.dimension)
    )
    expected = raw + raw.conjugate().T
    operator = _operator_from_modal_matrix(space, expected)
    mapper = OneHotVibrationalMapper()
    mapped_operator = mapper.map_operator(operator)
    mapped = mapped_operator.to_matrix()
    isometry = space.one_hot_isometry()

    assert operator.is_hermitian(atol=1e-14)
    assert mapped_operator.is_hermitian(atol=1e-14)
    assert operator.preserves_one_hot_subspace()
    assert mapper.num_qubits(space) == sum(num_modals)
    np.testing.assert_allclose(
        operator.to_modal_matrix(),
        expected,
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        isometry.conjugate().T @ mapped @ isometry,
        expected,
        rtol=0.0,
        atol=1e-13,
    )
    # This stronger intertwining relation also proves that the mapped
    # Hamiltonian does not leak these legal operators out of the one-hot space.
    np.testing.assert_allclose(
        mapped @ isometry,
        isometry @ expected,
        rtol=0.0,
        atol=1e-13,
    )


def test_mapping_preserves_operator_composition_and_adjoint_on_full_register():
    space = ModalSpace((2, 2))
    left = VibrationalOp.from_terms(
        space,
        (
            (
                (
                    VibrationalLadder("+", 0, 1),
                    VibrationalLadder("-", 1, 0),
                ),
                0.7 - 0.2j,
            ),
        ),
    )
    right = VibrationalOp.from_terms(
        space,
        (
            (
                (
                    VibrationalLadder("-", 0, 1),
                    VibrationalLadder("+", 1, 0),
                ),
                -0.3 + 0.5j,
            ),
        ),
    )
    mapper = OneHotVibrationalMapper()
    left_matrix = mapper.map_operator(left).to_matrix()
    right_matrix = mapper.map_operator(right).to_matrix()

    np.testing.assert_allclose(
        mapper.map_operator(left @ right).to_matrix(),
        left_matrix @ right_matrix,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        mapper.map_operator(left.adjoint()).to_matrix(),
        left_matrix.conjugate().T,
        rtol=0.0,
        atol=1e-14,
    )


def test_zero_and_identity_map_to_valid_core_hamiltonians():
    space = ModalSpace((2, 1))
    mapper = OneHotVibrationalMapper()
    zero = mapper.map_operator(VibrationalOp.zero(space))
    identity = mapper.map_operator(VibrationalOp.identity(space))

    assert zero.num_qubits == 3
    assert zero.terms == []
    np.testing.assert_array_equal(zero.to_matrix(), np.zeros((8, 8)))
    np.testing.assert_array_equal(identity.to_matrix(), np.eye(8))


def test_mapper_rejects_wrong_input_types():
    mapper = OneHotVibrationalMapper()
    with pytest.raises(TypeError, match="operator must be"):
        mapper.map_operator(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="modal_space must be"):
        mapper.num_qubits(object())  # type: ignore[arg-type]
