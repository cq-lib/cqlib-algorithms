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
