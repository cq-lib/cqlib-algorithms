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

"""Independent matrix and sector tests for lattice variational circuits."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from cqlib_algorithms.extension.lattice import (
    FermiHubbardModel,
    FermionicLatticeConfiguration,
    FermionicSector,
    HeisenbergModel,
    IsingModel,
    Lattice,
    LatticeSystem,
    LineLattice,
    SpinLatticeConfiguration,
    SpinMagnetizationSector,
)
from cqlib_algorithms.operators.ansatz import (
    FermiHubbardAnsatz,
    FermionicHoppingAnsatz,
    SpinExchangeAnsatz,
)
from cqlib_algorithms.protocols import Ansatz

_ATOL = 1e-12


def _annihilation(num_modes: int, mode: int) -> NDArray[np.complex128]:
    dimension = 1 << num_modes
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    lower_mask = (1 << mode) - 1
    for basis_index in range(dimension):
        if (basis_index >> mode) & 1:
            target = basis_index ^ (1 << mode)
            sign = -1.0 if (basis_index & lower_mask).bit_count() % 2 else 1.0
            result[target, basis_index] = sign
    return result


def _hopping_matrix(
    annihilators: list[NDArray[np.complex128]],
    source: int,
    target: int,
    hopping: complex,
) -> NDArray[np.complex128]:
    source_op = annihilators[source]
    target_op = annihilators[target]
    return (
        hopping * source_op.conj().T @ target_op
        + hopping.conjugate() * target_op.conj().T @ source_op
    )


def _basis_preparation_matrix(num_qubits: int, basis_index: int):
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        matrix[column ^ basis_index, column] = 1.0
    return matrix


def _number_matrix(
    annihilators: list[NDArray[np.complex128]],
    modes: tuple[int, ...],
) -> NDArray[np.complex128]:
    dimension = annihilators[0].shape[0]
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for mode in modes:
        operator = annihilators[mode]
        result += operator.conj().T @ operator
    return result


@pytest.mark.reference
@pytest.mark.parametrize(
    ("num_modes", "source", "target", "hopping", "theta"),
    [
        (2, 0, 1, 0.8, 0.37),
        (4, 0, 3, 0.7 + 0.4j, -0.23),
        (4, 3, 0, -0.5 + 0.2j, 0.61),
        (5, 1, 4, -0.9j, -0.44),
    ],
)
def test_fermionic_hopping_matches_independent_fock_space_exponential(
    num_modes: int,
    source: int,
    target: int,
    hopping: complex,
    theta: float,
):
    component = FermionicHoppingAnsatz(
        num_modes,
        source,
        target,
        hopping,
    )
    circuit = component.build_circuit("hop")
    actual = circuit.assign_parameters({"hop_0": theta}).to_matrix()
    annihilators = [_annihilation(num_modes, mode) for mode in range(num_modes)]
    generator = _hopping_matrix(annihilators, source, target, hopping)
    expected = expm(-1j * theta * generator)

    np.testing.assert_allclose(actual, expected, rtol=_ATOL, atol=_ATOL)
    np.testing.assert_allclose(
        actual.conj().T @ actual,
        np.eye(1 << num_modes),
        rtol=_ATOL,
        atol=_ATOL,
    )
    assert circuit.used_symbols == ["hop_0"]
    assert component.num_parameters() == 1
    assert component.num_qubits() == num_modes
    assert component.modes == (source, target)
    assert component.hopping == hopping
    assert isinstance(component, Ansatz)


@pytest.mark.reference
def test_fermionic_hopping_preserves_each_spin_population():
    component = FermionicHoppingAnsatz(6, 0, 4, -0.3 + 0.8j)
    unitary = component.build_circuit("t").assign_parameters({"t_0": 0.71}).to_matrix()
    annihilators = [_annihilation(6, mode) for mode in range(6)]
    number_up = _number_matrix(annihilators, (0, 2, 4))
    number_down = _number_matrix(annihilators, (1, 3, 5))

    np.testing.assert_allclose(
        unitary @ number_up,
        number_up @ unitary,
        rtol=_ATOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        unitary @ number_down,
        number_down @ unitary,
        rtol=_ATOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((True, 0, 1, 1.0), TypeError, "num_qubits"),
        ((2, True, 1, 1.0), TypeError, "source_mode"),
        ((2, 0, 2, 1.0), ValueError, "target_mode"),
        ((2, 0, 0, 1.0), ValueError, "distinct"),
        ((2, 0, 1, True), TypeError, "hopping"),
        ((2, 0, 1, 0.0), ValueError, "nonzero"),
        ((2, 0, 1, complex(math.inf, 0.0)), ValueError, "finite"),
    ],
)
def test_fermionic_hopping_rejects_invalid_input(args, error, message: str):
    with pytest.raises(error, match=message):
        FermionicHoppingAnsatz(*args)


def _hubbard_fixture() -> tuple[
    FermiHubbardModel,
    LatticeSystem,
    FermionicLatticeConfiguration,
]:
    lattice = Lattice.from_nodes_and_edges(
        2,
        (
            (0, 0, 0.3),
            (0, 1, -0.7 + 0.2j),
            (1, 1, -0.1),
        ),
    )
    model = FermiHubbardModel(lattice, onsite_interaction=(1.1, -0.4))
    system = LatticeSystem(model, FermionicSector(2, 1, 1))
    configuration = FermionicLatticeConfiguration(2, (0,), (1,))
    return model, system, configuration


@pytest.mark.reference
def test_fermi_hubbard_ansatz_matches_ordered_independent_exponentials():
    model, system, configuration = _hubbard_fixture()
    component = FermiHubbardAnsatz(system, configuration)
    circuit = component.build_circuit("fh")
    values = (0.21, -0.31, 0.17)
    actual = circuit.assign_parameters(
        dict(zip(circuit.used_symbols, values, strict=True))
    ).to_matrix()

    num_modes = model.num_modes
    annihilators = [_annihilation(num_modes, mode) for mode in range(num_modes)]
    hopping_up = _hopping_matrix(annihilators, 0, 2, -0.7 + 0.2j)
    hopping_down = _hopping_matrix(annihilators, 1, 3, -0.7 + 0.2j)
    numbers = [operator.conj().T @ operator for operator in annihilators]
    diagonal = (
        0.3 * (numbers[0] + numbers[1])
        - 0.1 * (numbers[2] + numbers[3])
        + 1.1 * numbers[0] @ numbers[1]
        - 0.4 * numbers[2] @ numbers[3]
    )
    reference = _basis_preparation_matrix(
        num_modes,
        configuration.basis_index,
    )
    expected = (
        expm(-1j * values[2] * diagonal)
        @ expm(-1j * values[1] * hopping_down)
        @ expm(-1j * values[0] * hopping_up)
        @ reference
    )

    np.testing.assert_allclose(actual, expected, rtol=_ATOL, atol=_ATOL)
    assert circuit.used_symbols == ["fh_0", "fh_1", "fh_2"]
    assert component.num_parameters() == 3
    assert component.num_qubits() == 4
    assert component.system is system
    assert component.configuration is configuration


@pytest.mark.reference
def test_fermi_hubbard_multiple_edges_keep_noncommuting_factor_order():
    hopping = 0.4 + 0.3j
    model = FermiHubbardModel(
        LineLattice(3, edge_parameter=hopping),
        onsite_interaction=0.0,
    )
    system = LatticeSystem(model, FermionicSector(3, 2, 1))
    configuration = FermionicLatticeConfiguration(3, (0, 2), (1,))
    component = FermiHubbardAnsatz(system, configuration)
    circuit = component.build_circuit("fh")
    values = (0.11, -0.23, 0.37, -0.41)
    actual = circuit.assign_parameters(
        dict(zip(circuit.used_symbols, values, strict=True))
    ).to_matrix()

    annihilators = [_annihilation(6, mode) for mode in range(6)]
    factors = (
        _hopping_matrix(annihilators, 0, 2, hopping),
        _hopping_matrix(annihilators, 1, 3, hopping),
        _hopping_matrix(annihilators, 2, 4, hopping),
        _hopping_matrix(annihilators, 3, 5, hopping),
    )
    expected = _basis_preparation_matrix(6, configuration.basis_index)
    for theta, factor in zip(values, factors, strict=True):
        expected = expm(-1j * theta * factor) @ expected

    np.testing.assert_allclose(actual, expected, rtol=_ATOL, atol=_ATOL)
    assert circuit.used_symbols == ["fh_0", "fh_1", "fh_2", "fh_3"]


@pytest.mark.reference
def test_fermi_hubbard_ansatz_has_zero_sector_leakage_for_multiple_layers():
    _, system, configuration = _hubbard_fixture()
    component = FermiHubbardAnsatz(system, configuration, reps=2)
    circuit = component.build_circuit("fh")
    bindings = {
        name: -0.41 + 0.13 * index for index, name in enumerate(circuit.used_symbols)
    }
    state = circuit.assign_parameters(bindings).to_matrix()[:, 0]

    for basis_index, amplitude in enumerate(state):
        num_up = sum((basis_index >> mode) & 1 for mode in (0, 2))
        num_down = sum((basis_index >> mode) & 1 for mode in (1, 3))
        if (num_up, num_down) != (1, 1):
            assert abs(amplitude) <= _ATOL
    assert circuit.used_symbols == [f"fh_{index}" for index in range(6)]
    assert component.num_parameters() == 6


def test_fermi_hubbard_zero_repetitions_and_zero_model_return_reference():
    model = FermiHubbardModel(Lattice.from_nodes_and_edges(1, ()))
    system = LatticeSystem(model, FermionicSector(1, 1, 0))
    configuration = FermionicLatticeConfiguration(1, (0,), ())

    for reps in (0, 1):
        component = FermiHubbardAnsatz(system, configuration, reps=reps)
        circuit = component.build_circuit("fh")
        assert circuit.used_symbols == []
        assert component.num_parameters() == 0
        expected = np.array((0.0, 1.0, 0.0, 0.0), dtype=np.complex128)
        np.testing.assert_allclose(circuit.to_matrix()[:, 0], expected, atol=0.0)


def test_fermi_hubbard_ansatz_rejects_incompatible_inputs():
    _, system, configuration = _hubbard_fixture()
    spin_system = LatticeSystem(
        HeisenbergModel(
            LineLattice(2, edge_parameter=1.0),
            coupling=(1.0, 1.0, 0.5),
        ),
        SpinMagnetizationSector(2, 1),
    )

    with pytest.raises(TypeError, match="LatticeSystem"):
        FermiHubbardAnsatz(object(), configuration)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FermiHubbardModel"):
        FermiHubbardAnsatz(spin_system, configuration)
    with pytest.raises(TypeError, match="FermionicLatticeConfiguration"):
        FermiHubbardAnsatz(
            system,
            SpinLatticeConfiguration(2, (0,)),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="system sector"):
        FermiHubbardAnsatz(
            system,
            FermionicLatticeConfiguration(2, (0,), ()),
        )
    with pytest.raises(TypeError, match="reps"):
        FermiHubbardAnsatz(system, configuration, reps=True)


def _spin_system(*, with_sector: bool = True) -> LatticeSystem:
    model = HeisenbergModel(
        LineLattice(3, edge_parameter=1.0),
        coupling=(1.0, 1.0, 0.7),
        external_field=(0.0, 0.0, 0.2),
    )
    sector = SpinMagnetizationSector(3, 2) if with_sector else None
    return LatticeSystem(model, sector)


@pytest.mark.reference
@pytest.mark.parametrize("mode", ["iswap", "fsim"])
def test_spin_exchange_ansatz_preserves_fixed_magnetization(mode: str):
    system = _spin_system()
    configuration = SpinLatticeConfiguration.neel(3)
    component = SpinExchangeAnsatz(
        system,
        configuration,
        reps=2,
        mode=mode,  # type: ignore[arg-type]
    )
    circuit = component.build_circuit("spin")
    bindings = {
        name: 0.09 * (index + 1) for index, name in enumerate(circuit.used_symbols)
    }
    state = circuit.assign_parameters(bindings).to_matrix()[:, 0]

    for basis_index, amplitude in enumerate(state):
        num_down = basis_index.bit_count()
        if num_down != system.sector.num_down:  # type: ignore[union-attr]
            assert abs(amplitude) <= _ATOL
    assert component.system is system
    assert component.configuration is configuration
    assert component.num_qubits() == 3
    assert component.num_parameters() == len(circuit.used_symbols)


def test_spin_exchange_zero_parameters_recover_reference_and_reps_zero_is_fixed():
    system = _spin_system()
    configuration = SpinLatticeConfiguration.neel(3)
    active = SpinExchangeAnsatz(system, configuration, reps=1)
    active_circuit = active.build_circuit("s")
    state = active_circuit.assign_parameters(
        {name: 0.0 for name in active_circuit.used_symbols}
    ).to_matrix()[:, 0]
    expected = np.zeros(8, dtype=np.complex128)
    expected[configuration.basis_index] = 1.0

    np.testing.assert_allclose(state, expected, rtol=_ATOL, atol=_ATOL)
    inactive = SpinExchangeAnsatz(system, configuration, reps=0)
    assert inactive.num_parameters() == 0
    assert inactive.build_circuit("s").used_symbols == []


def test_spin_exchange_rejects_missing_or_incompatible_sector():
    configuration = SpinLatticeConfiguration.neel(3)
    with pytest.raises(TypeError, match="fixed SpinMagnetizationSector"):
        SpinExchangeAnsatz(_spin_system(with_sector=False), configuration)

    fermi_model = FermiHubbardModel(LineLattice(2, edge_parameter=-1.0))
    fermi_system = LatticeSystem(fermi_model, FermionicSector(2, 1, 1))
    with pytest.raises(TypeError, match="spin model"):
        SpinExchangeAnsatz(
            fermi_system,
            SpinLatticeConfiguration(2, (0,)),
        )

    spin_system = _spin_system()
    with pytest.raises(ValueError, match="system sector"):
        SpinExchangeAnsatz(
            spin_system,
            SpinLatticeConfiguration(3, (0,)),
        )
    with pytest.raises(TypeError, match="SpinLatticeConfiguration"):
        SpinExchangeAnsatz(
            spin_system,
            FermionicLatticeConfiguration(3, (0, 2), (1,)),  # type: ignore[arg-type]
        )


def test_spin_exchange_rejects_nonconserving_model_at_system_boundary():
    model = IsingModel(
        LineLattice(2, edge_parameter=1.0),
        transverse_field=0.5,
    )
    with pytest.raises(ValueError, match="conserves total S_z"):
        LatticeSystem(model, SpinMagnetizationSector(2, 1))
