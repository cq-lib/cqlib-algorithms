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

"""Immutable weighted lattice topologies."""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from itertools import product
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._numeric import normalize_finite_complex

WeightedEdge: TypeAlias = tuple[int, int, complex]

_HERMITIAN_ATOL = 1e-12


class BoundaryCondition(Enum):
    """Boundary condition along one lattice direction."""

    OPEN = "open"
    PERIODIC = "periodic"


def _normalize_positive_int(value: int, *, name: str) -> int:
    """Validate and return a positive integer."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _normalize_node_index(value: int, *, name: str, num_nodes: int) -> int:
    """Validate and return an in-range node index."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0 or value >= num_nodes:
        raise ValueError(f"{name} must be in [0, num_nodes)")
    return value


def _normalize_real_weight(value: complex, *, name: str) -> complex:
    """Validate and return a finite real weight as a complex value."""
    result = normalize_finite_complex(value, name=name)
    if result.imag != 0.0:
        raise ValueError(f"{name} must be real")
    return complex(result.real)


def _copy_weighted_edges(
    num_nodes: int,
    weighted_edges: Iterable[WeightedEdge],
) -> tuple[WeightedEdge, ...]:
    try:
        raw_edges = tuple(weighted_edges)
    except TypeError as exc:
        raise TypeError("weighted_edges must be iterable") from exc

    normalized: list[WeightedEdge] = []
    seen: set[tuple[int, int]] = set()
    for position, edge in enumerate(raw_edges):
        if isinstance(edge, (str, bytes)) or not isinstance(edge, Sequence):
            raise TypeError(f"weighted_edges[{position}] must be a three-item sequence")
        if len(edge) != 3:
            raise ValueError(
                f"weighted_edges[{position}] must contain exactly three items"
            )

        source = _normalize_node_index(
            edge[0],
            name=f"weighted_edges[{position}][0]",
            num_nodes=num_nodes,
        )
        target = _normalize_node_index(
            edge[1],
            name=f"weighted_edges[{position}][1]",
            num_nodes=num_nodes,
        )
        weight = normalize_finite_complex(
            edge[2],
            name=f"weighted_edges[{position}][2]",
        )
        if source == target:
            weight = _normalize_real_weight(
                weight,
                name=f"weighted_edges[{position}][2] for a self-loop",
            )
        elif source > target:
            source, target = target, source
            weight = weight.conjugate()

        pair = (source, target)
        if pair in seen:
            raise ValueError(f"duplicate undirected edge {pair}")
        seen.add(pair)
        normalized.append((source, target, weight))

    normalized.sort(key=lambda item: (item[0], item[1]))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class Lattice:
    """Finite undirected lattice with Hermitian complex edge weights.

    Each edge is stored once in upper-triangular orientation. Reversing an
    input edge conjugates its weight, so the resulting adjacency matrix is
    Hermitian. Self-loop weights must be real.

    Args:
        num_nodes: Positive number of numbered lattice nodes.
        weighted_edges: Unique finite weighted undirected edges.

    Raises:
        TypeError: If a count, edge, or weight has the wrong type.
        ValueError: If indices, duplicate edges, or weights are invalid.
    """

    num_nodes: int
    weighted_edges: tuple[WeightedEdge, ...]

    def __post_init__(self) -> None:
        num_nodes = _normalize_positive_int(self.num_nodes, name="num_nodes")
        edges = _copy_weighted_edges(num_nodes, self.weighted_edges)
        object.__setattr__(self, "weighted_edges", edges)

    @classmethod
    def from_nodes_and_edges(
        cls,
        num_nodes: int,
        weighted_edges: Iterable[WeightedEdge],
    ) -> Lattice:
        """Construct a lattice from numbered nodes and weighted edges."""
        return cls(num_nodes=num_nodes, weighted_edges=tuple(weighted_edges))

    @classmethod
    def from_adjacency_matrix(cls, matrix: ArrayLike) -> Lattice:
        """Construct a lattice from a finite Hermitian adjacency matrix."""
        try:
            raw = np.asarray(matrix)
        except (TypeError, ValueError) as exc:
            raise TypeError("matrix must be a numeric array-like object") from exc
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
            raise ValueError("matrix must be a non-empty square matrix")
        if raw.shape[0] == 0:
            raise ValueError("matrix must be a non-empty square matrix")
        if raw.dtype.kind not in "biufc":
            raise TypeError("matrix must contain numeric values")

        adjacency = np.array(raw, dtype=np.complex128, copy=True)
        if not np.all(np.isfinite(adjacency)):
            raise ValueError("matrix must contain only finite values")
        if not np.allclose(
            adjacency,
            adjacency.conjugate().T,
            rtol=0.0,
            atol=_HERMITIAN_ATOL,
        ):
            raise ValueError("matrix must be Hermitian")

        adjacency = 0.5 * (adjacency + adjacency.conjugate().T)
        edges: list[WeightedEdge] = []
        for source in range(adjacency.shape[0]):
            for target in range(source, adjacency.shape[1]):
                weight = complex(adjacency[source, target])
                if weight != 0.0:
                    edges.append((source, target, weight))
        return cls.from_nodes_and_edges(adjacency.shape[0], edges)

    @property
    def node_indices(self) -> tuple[int, ...]:
        """Return the ordered node labels."""
        return tuple(range(self.num_nodes))

    @property
    def num_edges(self) -> int:
        """Return the number of declared edges, including self-loops."""
        return len(self.weighted_edges)

    def edge_weight(self, source: int, target: int) -> complex | None:
        """Return an oriented edge weight, or ``None`` when no edge exists."""
        source = _normalize_node_index(source, name="source", num_nodes=self.num_nodes)
        target = _normalize_node_index(target, name="target", num_nodes=self.num_nodes)
        reverse = source > target
        if reverse:
            source, target = target, source
        for left, right, weight in self.weighted_edges:
            if left == source and right == target:
                return weight.conjugate() if reverse else weight
        return None

    def to_adjacency_matrix(self, *, weighted: bool = False) -> NDArray[np.complex128]:
        """Return a binary or weighted Hermitian adjacency matrix.

        The binary form records declared connectivity, including zero-weight
        edges and declared self-loops. The weighted form records physical edge
        weights and is therefore zero at a declared zero-weight edge.
        """
        if type(weighted) is not bool:
            raise TypeError("weighted must be a bool")
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.complex128)
        for source, target, weight in self.weighted_edges:
            value = weight if weighted else 1.0 + 0.0j
            matrix[source, target] = value
            matrix[target, source] = value.conjugate()
        return matrix

    def uniform_parameters(
        self,
        interaction: complex,
        onsite_potential: complex,
    ) -> Lattice:
        """Return the same connectivity with uniform physical parameters."""
        interaction = normalize_finite_complex(interaction, name="interaction")
        onsite_potential = _normalize_real_weight(
            onsite_potential,
            name="onsite_potential",
        )
        edges = [
            (source, target, interaction)
            for source, target, _ in self.weighted_edges
            if source != target
        ]
        edges.extend((node, node, onsite_potential) for node in range(self.num_nodes))
        return Lattice.from_nodes_and_edges(self.num_nodes, edges)


def _normalize_size(size: Sequence[int]) -> tuple[int, ...]:
    if isinstance(size, (str, bytes)) or not isinstance(size, Sequence):
        raise TypeError("size must be a non-empty sequence of ints")
    copied = tuple(size)
    if not copied:
        raise ValueError("size must contain at least one dimension")
    return tuple(
        _normalize_positive_int(extent, name=f"size[{axis}]")
        for axis, extent in enumerate(copied)
    )


def _normalize_edge_parameters(
    value: complex | Sequence[complex],
    dimension: int,
) -> tuple[complex, ...]:
    if isinstance(value, numbers.Complex) and not isinstance(value, (bool, np.bool_)):
        parameter = normalize_finite_complex(complex(value), name="edge_parameter")
        return (parameter,) * dimension
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("edge_parameter must be a number or a sequence of numbers")
    copied = tuple(value)
    if len(copied) != dimension:
        raise ValueError("edge_parameter length must match the lattice dimension")
    return tuple(
        normalize_finite_complex(parameter, name=f"edge_parameter[{axis}]")
        for axis, parameter in enumerate(copied)
    )


def _normalize_boundary_conditions(
    value: BoundaryCondition | Sequence[BoundaryCondition],
    dimension: int,
) -> tuple[BoundaryCondition, ...]:
    if isinstance(value, BoundaryCondition):
        return (value,) * dimension
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "boundary_condition must be a BoundaryCondition or a sequence of them"
        )
    copied = tuple(value)
    if len(copied) != dimension:
        raise ValueError("boundary_condition length must match the lattice dimension")
    for axis, condition in enumerate(copied):
        if not isinstance(condition, BoundaryCondition):
            raise TypeError(f"boundary_condition[{axis}] must be a BoundaryCondition")
    return copied


class HyperCubicLattice(Lattice):
    """Nearest-neighbor lattice in an arbitrary number of dimensions.

    The topology is a simple graph with at most one undirected edge per node
    pair. Consequently a periodic axis of extent two has the same connectivity
    as its open counterpart; the representation does not duplicate the two
    directed nearest-neighbor displacements as parallel edges.

    Args:
        size: Positive extent of each dimension, with axis zero fastest.
        edge_parameter: Uniform or per-axis finite nearest-neighbor weights.
        onsite_parameter: Finite real self-loop weight at every node.
        boundary_condition: Uniform or per-axis boundary conditions.

    Raises:
        TypeError: If a size, weight, or boundary value has the wrong type.
        ValueError: If dimensions, values, or sequence lengths are invalid.

    Warns:
        RuntimeWarning: If a periodic axis has extent two and therefore uses
            the same simple-graph edge as its open counterpart.
    """

    __slots__ = (
        "_boundary_conditions",
        "_edge_parameters",
        "_onsite_parameter",
        "_size",
    )

    _boundary_conditions: tuple[BoundaryCondition, ...]
    _edge_parameters: tuple[complex, ...]
    _onsite_parameter: complex
    _size: tuple[int, ...]

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __init__(
        self,
        size: Sequence[int],
        *,
        edge_parameter: complex | Sequence[complex] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition
        | Sequence[BoundaryCondition] = BoundaryCondition.OPEN,
    ) -> None:
        normalized_size = _normalize_size(size)
        dimension = len(normalized_size)
        edge_parameters = _normalize_edge_parameters(edge_parameter, dimension)
        boundary_conditions = _normalize_boundary_conditions(
            boundary_condition,
            dimension,
        )
        onsite = _normalize_real_weight(onsite_parameter, name="onsite_parameter")

        extent_two_periodic_axes = tuple(
            axis
            for axis, (extent, condition) in enumerate(
                zip(normalized_size, boundary_conditions, strict=True)
            )
            if extent == 2 and condition is BoundaryCondition.PERIODIC
        )
        if extent_two_periodic_axes:
            warnings.warn(
                "periodic axis indices "
                f"{extent_two_periodic_axes} have extent two; simple-graph "
                "semantics do not add parallel wrap edges, so these axes have "
                "the same connectivity and edge weights as open boundaries",
                RuntimeWarning,
                stacklevel=2,
            )

        edges: list[WeightedEdge] = []
        coordinates = tuple(product(*(range(extent) for extent in normalized_size)))
        for coordinate in coordinates:
            source = self._coordinate_to_index_unchecked(coordinate, normalized_size)
            for axis, extent in enumerate(normalized_size):
                if coordinate[axis] + 1 < extent:
                    neighbor = list(coordinate)
                    neighbor[axis] += 1
                    target = self._coordinate_to_index_unchecked(
                        tuple(neighbor),
                        normalized_size,
                    )
                    edges.append((source, target, edge_parameters[axis]))
                elif (
                    boundary_conditions[axis] is BoundaryCondition.PERIODIC
                    and extent > 2
                ):
                    neighbor = list(coordinate)
                    neighbor[axis] = 0
                    target = self._coordinate_to_index_unchecked(
                        tuple(neighbor),
                        normalized_size,
                    )
                    edges.append((source, target, edge_parameters[axis]))

        num_nodes = math.prod(normalized_size)
        edges.extend((node, node, onsite) for node in range(num_nodes))
        super().__init__(num_nodes=num_nodes, weighted_edges=tuple(edges))
        object.__setattr__(self, "_size", normalized_size)
        object.__setattr__(self, "_edge_parameters", edge_parameters)
        object.__setattr__(self, "_onsite_parameter", onsite)
        object.__setattr__(self, "_boundary_conditions", boundary_conditions)

    @staticmethod
    def _coordinate_to_index_unchecked(
        coordinate: tuple[int, ...],
        size: tuple[int, ...],
    ) -> int:
        index = 0
        stride = 1
        for component, extent in zip(coordinate, size, strict=True):
            index += component * stride
            stride *= extent
        return index

    @property
    def dimension(self) -> int:
        """Return the number of lattice dimensions."""
        return len(self._size)

    @property
    def size(self) -> tuple[int, ...]:
        """Return the length of each lattice dimension."""
        return self._size

    @property
    def edge_parameters(self) -> tuple[complex, ...]:
        """Return the edge weight in each positive coordinate direction."""
        return self._edge_parameters

    @property
    def onsite_parameter(self) -> complex:
        """Return the uniform self-loop weight."""
        return self._onsite_parameter

    @property
    def boundary_conditions(self) -> tuple[BoundaryCondition, ...]:
        """Return the boundary condition in each coordinate direction."""
        return self._boundary_conditions

    def coordinate_to_index(self, coordinate: Sequence[int]) -> int:
        """Convert a coordinate to its node index, with axis zero fastest."""
        if isinstance(coordinate, (str, bytes)) or not isinstance(coordinate, Sequence):
            raise TypeError("coordinate must be a sequence of ints")
        copied = tuple(coordinate)
        if len(copied) != self.dimension:
            raise ValueError("coordinate length must match the lattice dimension")
        for axis, (component, extent) in enumerate(
            zip(copied, self._size, strict=True)
        ):
            if type(component) is not int:
                raise TypeError(f"coordinate[{axis}] must be an int")
            if component < 0 or component >= extent:
                raise ValueError(f"coordinate[{axis}] is out of range")
        return self._coordinate_to_index_unchecked(copied, self._size)

    def index_to_coordinate(self, index: int) -> tuple[int, ...]:
        """Convert a node index to a coordinate, with axis zero fastest."""
        index = _normalize_node_index(index, name="index", num_nodes=self.num_nodes)
        components: list[int] = []
        remainder = index
        for extent in self._size:
            components.append(remainder % extent)
            remainder //= extent
        return tuple(components)


class LineLattice(HyperCubicLattice):
    """One-dimensional nearest-neighbor lattice.

    Args:
        num_nodes: Positive number of line sites.
        edge_parameter: Finite nearest-neighbor weight.
        onsite_parameter: Finite real onsite weight.
        boundary_condition: Open or periodic boundary condition.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the size or a weight is invalid.
    """

    __slots__ = ()

    def __init__(
        self,
        num_nodes: int,
        *,
        edge_parameter: complex = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.OPEN,
    ) -> None:
        super().__init__(
            (_normalize_positive_int(num_nodes, name="num_nodes"),),
            edge_parameter=edge_parameter,
            onsite_parameter=onsite_parameter,
            boundary_condition=boundary_condition,
        )

    @property
    def edge_parameter(self) -> complex:
        """Return the nearest-neighbor edge weight."""
        return self.edge_parameters[0]

    @property
    def boundary_condition(self) -> BoundaryCondition:
        """Return the line boundary condition."""
        return self.boundary_conditions[0]


class SquareLattice(HyperCubicLattice):
    """Two-dimensional rectangular nearest-neighbor lattice.

    Args:
        rows: Positive axis-zero extent.
        cols: Positive axis-one extent.
        edge_parameter: Uniform or per-axis finite edge weights.
        onsite_parameter: Finite real onsite weight.
        boundary_condition: Uniform or per-axis boundary conditions.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions, values, or sequence lengths are invalid.
    """

    __slots__ = ()

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        edge_parameter: complex | Sequence[complex] = 1.0,
        onsite_parameter: complex = 0.0,
        boundary_condition: BoundaryCondition
        | Sequence[BoundaryCondition] = BoundaryCondition.OPEN,
    ) -> None:
        super().__init__(
            (
                _normalize_positive_int(rows, name="rows"),
                _normalize_positive_int(cols, name="cols"),
            ),
            edge_parameter=edge_parameter,
            onsite_parameter=onsite_parameter,
            boundary_condition=boundary_condition,
        )

    @property
    def rows(self) -> int:
        """Return the number of sites along axis zero."""
        return self.size[0]

    @property
    def cols(self) -> int:
        """Return the number of sites along axis one."""
        return self.size[1]
