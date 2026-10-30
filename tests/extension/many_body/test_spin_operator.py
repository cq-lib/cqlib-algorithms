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

"""Tests for the temporary physical-spin intermediate representation."""

from fractions import Fraction

import numpy as np
import pytest

from cqlib_algorithms.extension.many_body import SpinHalfMapper, SpinOp

X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.diag([1.0, -1.0]).astype(np.complex128)
I = np.eye(2, dtype=np.complex128)


def test_spin_half_single_site_uses_physical_angular_momentum_normalization():
    operator = SpinOp.from_terms(
        1,
        (
            ((("X", 0),), 2.0),
            ((("Y", 0),), -4.0),
            ((("Z", 0),), 6.0),
        ),
    )

    expected = X - 2.0 * Y + 3.0 * Z
    np.testing.assert_allclose(operator.to_matrix(), expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        SpinHalfMapper().map_operator(operator).to_matrix(),
        expected,
        rtol=0.0,
        atol=1e-14,
    )


def test_spin_half_mapper_preserves_same_site_product_order_and_phase():
    xy = SpinOp.from_terms(1, ((((("X", 0), ("Y", 0))), 1.0),))
    yx = SpinOp.from_terms(1, ((((("Y", 0), ("X", 0))), 1.0),))

    np.testing.assert_allclose(xy.to_matrix(), 0.25j * Z, atol=1e-14)
    np.testing.assert_allclose(yx.to_matrix(), -0.25j * Z, atol=1e-14)
    np.testing.assert_allclose(
        SpinHalfMapper().map_operator(xy).to_matrix(),
        xy.to_matrix(),
        atol=1e-14,
    )
    assert xy.adjoint().equiv(yx)
    assert not xy.is_hermitian()


def test_multi_site_matrix_uses_site_zero_as_least_significant_subsystem():
    operator = SpinOp.from_terms(
        2,
        (
            ((("X", 0),), 2.0),
            ((("Z", 1),), 4.0),
            ((("Y", 0), ("X", 1)), -8.0),
        ),
    )

    expected = np.kron(I, X) + 2.0 * np.kron(Z, I) - 2.0 * np.kron(X, Y)
    np.testing.assert_allclose(operator.to_matrix(), expected, atol=1e-14)


def test_spin_one_matrices_obey_angular_momentum_commutation_relation():
    sx = SpinOp.from_terms(1, ((((("X", 0),)), 1.0),), spin=1)
    sy = SpinOp.from_terms(1, ((((("Y", 0),)), 1.0),), spin=1)
    sz = SpinOp.from_terms(1, ((((("Z", 0),)), 1.0),), spin=1)

    commutator = sx @ sy - sy @ sx
    assert commutator.equiv(1.0j * sz, atol=1e-14)
    np.testing.assert_allclose(
        sx.to_matrix() @ sx.to_matrix()
        + sy.to_matrix() @ sy.to_matrix()
        + sz.to_matrix() @ sz.to_matrix(),
        2.0 * np.eye(3),
        atol=1e-14,
    )


def test_spin_half_equivalence_uses_sparse_mapping_without_dense_matrices(
    monkeypatch,
):
    num_spins = 64
    sx = SpinOp.from_terms(num_spins, (((("X", 0),), 1.0),))
    sy = SpinOp.from_terms(num_spins, (((("Y", 0),), 1.0),))
    sz = SpinOp.from_terms(num_spins, (((("Z", 0),), 1.0),))
    represented_sx = sx + (sx @ sy - 0.5j * sz)

    def fail_dense_conversion(self):
        raise AssertionError("spin-one-half equivalence must stay sparse")

    monkeypatch.setattr(SpinOp, "to_matrix", fail_dense_conversion)

    assert represented_sx.equiv(sx, atol=1e-14)
    assert represented_sx.is_hermitian(atol=1e-14)


def test_large_higher_spin_equivalence_stops_before_dense_allocation(monkeypatch):
    left = SpinOp.from_terms(7, (((("X", 0), ("Y", 0)), 1.0),), spin=1)
    right = 1.0j * SpinOp.from_terms(7, (((("Z", 0),), 1.0),), spin=1)

    def fail_dense_conversion(self):
        raise AssertionError("resource guard must run before dense conversion")

    monkeypatch.setattr(SpinOp, "to_matrix", fail_dense_conversion)

    with (
        pytest.warns(RuntimeWarning, match="dense comparison"),
        pytest.raises(MemoryError, match="estimated"),
    ):
        left.equiv(right)


def test_construction_merges_duplicate_terms_and_defensively_copies_input():
    mutable_term = [("Z", 1), ("X", 0)]
    entries = [(mutable_term, 2.0), ((("X", 0), ("Z", 1)), -0.5)]
    operator = SpinOp.from_terms(2, entries)
    mutable_term.clear()
    entries.clear()

    assert operator.terms == (((("X", 0), ("Z", 1)), 1.5 + 0.0j),)


@pytest.mark.parametrize(
    ("num_spins", "terms", "spin", "error"),
    [
        (True, (), 0.5, TypeError),
        (0, (), 0.5, ValueError),
        (1, (), True, TypeError),
        (1, (), 0.3, ValueError),
        (1, (), float("inf"), ValueError),
        (1, (((("A", 0),), 1.0),), 0.5, ValueError),
        (1, (((("X", -1),), 1.0),), 0.5, ValueError),
        (1, (((("X", 1),), 1.0),), 0.5, ValueError),
        (1, (((("X", 0),), float("nan")),), 0.5, ValueError),
        (1, (((("X", 0),), True),), 0.5, TypeError),
    ],
)
def test_invalid_spin_operator_inputs_are_rejected(num_spins, terms, spin, error):
    with pytest.raises(error):
        SpinOp.from_terms(num_spins, terms, spin=spin)


def test_spin_half_mapper_rejects_non_half_spin_and_wrong_input_type():
    mapper = SpinHalfMapper()
    with pytest.raises(ValueError, match="only spin one half"):
        mapper.map_operator(SpinOp.identity(1, spin=1))
    with pytest.raises(TypeError, match="operator must be"):
        mapper.map_operator(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(("value", "error"), [(True, TypeError), (0, ValueError)])
def test_spin_half_mapper_validates_width(value, error):
    with pytest.raises(error):
        SpinHalfMapper.num_qubits(value)


def test_spin_value_is_stored_exactly_as_a_fraction():
    assert SpinOp.identity(1, spin=1.5).spin == Fraction(3, 2)


def test_zero_operator_and_wrong_arithmetic_operand_have_defined_behavior():
    zero = SpinOp.zero(2)

    assert zero.is_zero()
    np.testing.assert_array_equal(zero.to_matrix(), np.zeros((4, 4)))
    with pytest.raises(TypeError, match="other must be"):
        _ = zero - object()  # type: ignore[operator]
    with pytest.raises(TypeError, match="terms must be iterable"):
        SpinOp.from_terms(1, 42)  # type: ignore[arg-type]
