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

"""Tests for computational-basis data encoding."""

import numpy as np
import pytest

from cqlib_algorithms.operators.state_preparation import BasisEncoding


def test_copies_bits_and_delegates_little_endian_encoding() -> None:
    bits = [True, False, True]
    encoding = BasisEncoding(bits)
    bits[:] = [False, False, False]
    expected = np.zeros(8, dtype=complex)
    expected[5] = 1.0

    encoding.validate()
    assert encoding.num_qubits() == 3
    assert encoding.num_parameters() == 0
    np.testing.assert_allclose(
        encoding.build_circuit("ignored").to_matrix()[:, 0],
        expected,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("bits", [[], "01", b"01", 1, None])
def test_rejects_invalid_bit_sequences(bits: object) -> None:
    error = ValueError if bits == [] else TypeError
    with pytest.raises(error):
        BasisEncoding(bits)  # type: ignore[arg-type]


@pytest.mark.parametrize("bits", [[0, 1], [True, 0], [False, None]])
def test_requires_actual_bool_items(bits: list[object]) -> None:
    with pytest.raises(TypeError, match="item in bits"):
        BasisEncoding(bits)  # type: ignore[arg-type]
