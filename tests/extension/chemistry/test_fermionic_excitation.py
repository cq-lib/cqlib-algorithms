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

"""Independent Fock-space tests for fermionic excitation generators."""

import numpy as np
import pytest
from cqlib.qis import JordanWignerMapper

from cqlib_algorithms.extension.chemistry import FermionicExcitation


def _apply_ladder(state: int, mode: int, *, create: bool) -> tuple[int, float] | None:
    occupied = bool(state & (1 << mode))
    if occupied == create:
        return None
    sign = -1.0 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1.0
    output = state | (1 << mode) if create else state & ~(1 << mode)
    return output, sign


def _transition_matrix(excitation: FermionicExcitation) -> np.ndarray:
    size = 1 << excitation.num_spin_orbitals
    transition = np.zeros((size, size), dtype=np.complex128)
    operations = tuple((True, target) for target in excitation.targets) + tuple(
        (False, source) for source in reversed(excitation.sources)
    )
    for basis in range(size):
        output = basis
        amplitude = 1.0
        for create, mode in reversed(operations):
            applied = _apply_ladder(output, mode, create=create)
            if applied is None:
                break
            output, sign = applied
            amplitude *= sign
        else:
            transition[output, basis] = amplitude
    return transition


def _number_and_spin_z(num_spin_orbitals: int) -> tuple[np.ndarray, np.ndarray]:
    size = 1 << num_spin_orbitals
    number = np.diag([basis.bit_count() for basis in range(size)])
    spin_z = np.diag(
        [
            0.5
            * (
                sum(
                    bool(basis & (1 << mode)) for mode in range(0, num_spin_orbitals, 2)
                )
                - sum(
                    bool(basis & (1 << mode)) for mode in range(1, num_spin_orbitals, 2)
                )
            )
            for basis in range(size)
        ]
    )
    return number, spin_z


@pytest.mark.reference
@pytest.mark.parametrize(
    "excitation",
    [
        FermionicExcitation(8, (0,), (4,)),
        FermionicExcitation(8, (0, 2), (4, 6)),
        FermionicExcitation(8, (0, 1), (4, 5)),
        FermionicExcitation(8, (1, 3), (5, 7)),
    ],
)
def test_core_generator_matches_independent_hermitian_fock_matrix(excitation):
    transition = _transition_matrix(excitation)
    expected = 1j * (transition - transition.conj().T)
    generator = excitation.to_fermionic_generator()
    actual = JordanWignerMapper().map_operator(generator).to_matrix()
    number, spin_z = _number_and_spin_z(excitation.num_spin_orbitals)
    assert generator.is_hermitian(atol=0.0)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(actual, actual.conj().T, rtol=0.0, atol=1e-12)
    assert np.allclose(actual @ number - number @ actual, 0.0, atol=1e-12)
    assert np.allclose(actual @ spin_z - spin_z @ actual, 0.0, atol=1e-12)


def test_generator_calls_return_independent_core_objects():
    excitation = FermionicExcitation(4, (0,), (2,))
    first = excitation.to_fermionic_generator()
    second = excitation.to_fermionic_generator()
    first.add_term((), 1.0)
    assert first is not second
    assert first.num_terms == second.num_terms + 1


@pytest.mark.parametrize(
    "arguments",
    [
        (4, (), (2,)),
        (4, (0,), ()),
        (4, (0,), (2, 3)),
        (4, (0, 1), (1, 2)),
        (4, (1, 0), (2, 3)),
        (4, (0, 0), (2, 3)),
        (4, (True,), (2,)),
        (4, (0,), (4,)),
        (0, (0,), (1,)),
    ],
)
def test_excitation_rejects_invalid_index_sets(arguments):
    with pytest.raises((TypeError, ValueError)):
        FermionicExcitation(*arguments)


@pytest.mark.parametrize("value", [True, 4.0, "4", None])
def test_excitation_rejects_non_integer_width(value):
    with pytest.raises(TypeError, match="int"):
        FermionicExcitation(value, (0,), (1,))  # type: ignore[arg-type]


def test_excitation_does_not_silently_convert_containers_or_sort():
    with pytest.raises(TypeError, match="tuple"):
        FermionicExcitation(4, [0], (2,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly increasing"):
        FermionicExcitation(4, (1, 0), (2, 3))
