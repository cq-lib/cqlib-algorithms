## QSVT Algorithm Interface Specification

### Scope

This specification defines the first implementation slice of quantum singular
value transformation under `operators.oracle`. The scope is intentionally
limited to:

- standard block encodings only (`Pi = PiTilde = |0><0|^a ⊗ I`)
- `cqlib` circuit construction only
- phase solving reused from the existing QSP implementation
- square explicit matrices supported through existing `UnitaryDilation`

This specification does not include:

- non-standard block encodings with distinct left/right projectors
- rectangular block-encoding-specific alternating projector dimensions
- new classical phase-solving methods
- direct training or optimization frameworks

### Mathematical Contract

QSVT is built on a block encoding satisfying

`A = PiTilde @ U @ Pi`

at the projected-subspace level, or in the standard block-encoding notation

`A / alpha = (⟨0|^a ⊗ I) U (|0⟩^a ⊗ I)`.

For the first implementation slice:

- `Pi = PiTilde = |0><0|^a ⊗ I`
- the implementation must still model both projectors explicitly in the QSVT
  construction, even if they are equal in the standard case
- later non-standard extensions must be able to replace either projector
  without redesigning the public `QSVT` interface

### Module Layout

Confirmed file placement under the current project structure:

- `src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/__init__.py`
- `src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/qsp.py`
- `src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/_phase_solver/`
- `src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/projector_controlled_phase.py`
- `src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/qsvt.py`

Test placement:

- `tests/operators/composite/test_qsvt.py`
- `tests/operators/composite/test_quantum_signal_processing.py`
- `tests/operators/test_package.py`

### Layering

The implementation must preserve this dependency direction:

`BlockEncoding -> QuantumSingularValueTransformation -> QuantumSignalProcessing phase solver`

Concretely:

- `block_encoding/` provides block-encoding operators such as `UnitaryDilation`
- `qsp.py` remains the single-qubit QSP reference/core and phase-solver entry
- `projector_controlled_phase.py` provides the QSVT projector-phase component
- `qsvt.py` provides the alternating-sequence assembly and external QSVT API

`qsvt.py` may depend on `BlockEncoding`, `Circuit`, and phase solving. It must
not introduce a second independent phase-solver stack.

### Public Interfaces

#### 1. `ProjectorControlledPhase`

File:
`src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/projector_controlled_phase.py`

Purpose:
Build the standard projector phase rotation
`exp(i * phi * (2 * Pi - I))` for a standard block encoding, where
`Pi = |0><0|^a ⊗ I`.

Public definition:

```python
@final
class ProjectorControlledPhase:
    def __init__(
        self,
        num_system_qubits: int,
        num_ancilla_qubits: int,
        *,
        phase: float,
    ) -> None: ...

    @property
    def phase(self) -> float: ...

    @property
    def num_system_qubits(self) -> int: ...

    @property
    def num_ancilla_qubits(self) -> int: ...

    @property
    def num_qubits(self) -> int: ...

    @property
    def system_register_indices(self) -> list[int]: ...

    @property
    def ancilla_register_indices(self) -> list[int]: ...

    def build_circuit(self) -> Circuit: ...
```

Required behavior:

- Local qubit layout is `[system..., ancilla...]`
- The projector is controlled by the ancilla register being all-zero
- `build_circuit()` returns a fresh circuit each time
- The built circuit matrix must equal the exact diagonal projector-phase matrix
  on the local ordering
- The class represents the projector phase associated with one specific
  projector, even in the first slice where `Pi = PiTilde`

Exceptions:

- `TypeError` for non-integer qubit counts or non-real phase
- `ValueError` for negative qubit counts or zero total width

#### 2. `QuantumSingularValueTransformation`

File:
`src/cqlib_algorithms/operators/composite/quantum_singular_value_transformation/qsvt.py`

Purpose:
Build the standard QSVT alternating sequence from a standard block encoding and
 a solved phase sequence.

Public definition:

```python
@final
class QuantumSingularValueTransformation:
    def __init__(
        self,
        block_encoding: BlockEncoding,
        phases: Sequence[float] | np.ndarray,
    ) -> None: ...

    @property
    def block_encoding(self) -> BlockEncoding: ...

    @property
    def phases(self) -> tuple[float, ...]: ...

    @property
    def right_projector(self) -> ProjectorControlledPhase: ...

    @property
    def left_projector(self) -> ProjectorControlledPhase: ...

    @property
    def num_qubits(self) -> int: ...

    def build_circuit(self) -> Circuit: ...

    def check_block_encoding(
        self,
        *,
        expected_matrix: np.ndarray | None = None,
        atol: float = 1e-10,
    ) -> float: ...

    @classmethod
    def from_chebyshev_coefficients(
        cls,
        block_encoding: BlockEncoding,
        coefficients: Sequence[float] | np.ndarray,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 300,
        gradient_tolerance: float = 1e-10,
    ) -> "QuantumSingularValueTransformation": ...
```

Required behavior:

- Accept only standard block encodings in the first version
- Reuse the existing QSP phase solver from `qsp.py` / `_phase_solver.py`
- Explicitly construct both projector-associated phase rotations:
  - right projector `Pi`
  - left projector `PiTilde`
- Build the alternating sequence using:
  - left/right projector phase rotations
  - `block_encoding.build_circuit()`
  - `block_encoding.build_adjoint_circuit()`
- Preserve the local qubit ordering already used by the block encoding
- Return a fresh circuit on each `build_circuit()` call
- Provide an explicit block-encoding check function rather than running the
  projected identity query on every construction or every `build_circuit()`:
  - semantically, `A = PiTilde U Pi`
  - for the first slice, this reduces to checking the standard top-left
    projected block implied by the existing `BlockEncoding` interface
- Expose the left/right projector objects as explicit QSVT state, even if they
  coincide in the standard case

Exceptions:

- `TypeError` if `block_encoding` is not a `BlockEncoding`, or if phases /
  solver controls have invalid types
- `ValueError` if the block encoding is not standard, if the projected
  block-encoding contract is not satisfied when explicitly checked, or if the
  phase sequence is empty
- `RuntimeError` propagated from phase solving when convergence fails

### Alternating-Sequence Contract

For the first implementation slice, the sequence is restricted to the standard
projector case. The assembly rule is:

- insert projector-phase operations associated with the left/right projectors
  and the provided phase sequence
- alternate `U` and `U†` from the block encoding
- use the existing real-QSP phase sequence as the classical input

This first implementation does not expose an alternative parity-specific public
API. The parity-specific ordering remains internal to `qsvt.py`.

### Internal Helpers

Internal helper functions are allowed inside `qsvt.py` and
`projector_controlled_phase.py`, but no additional public classes are required
for the first slice.

Recommended internal helper responsibilities:

- normalize and validate phase sequences
- construct the exact standard projector operators `Pi` and `PiTilde`
- construct the exact standard projector-phase diagonal reference matrices
- validate the projected block-encoding identity `A = PiTilde U Pi`
- assemble the ordered list of circuit segments for QSVT
- isolate parity-specific sequence ordering

### Core Data Structures

Preferred data structures:

- `BlockEncoding` for the encoded operator input
- `Circuit` for all emitted quantum programs
- `tuple[float, ...]` for normalized phase sequences
- `list[int]` for local register index properties
- `numpy.ndarray` only for reference-matrix checks in tests or internal exact
  matrix helpers

No new protocol or inheritance hierarchy is required beyond the existing
`BlockEncoding`.

### Validation and Test Contract

#### `test_projector_controlled_phase.py`

Must cover:

- exact diagonal matrix action for small widths
- all-zero ancilla projector semantics
- explicit consistency with the associated standard projector `Pi`
- fresh-circuit behavior
- composition onto non-contiguous qubits
- input validation

#### `test_quantum_singular_value_transformation.py`

Must cover:

- construction from an explicit `UnitaryDilation`
- explicit construction of left/right projector-phase components
- fresh-circuit behavior
- composition onto non-contiguous qubits if supported by the built circuit
- constructor validation
- rejection of incompatible block encodings that do not satisfy the standard
  projected form
- `from_chebyshev_coefficients()` reusing the QSP phase solver

First-slice numerical correctness target:

- validate against small matrix instances only
- start with 1-qubit system matrices block-encoded by `UnitaryDilation`
- compare the built circuit matrix against an independently assembled reference
  alternating sequence using explicit `Pi`, `PiTilde`, `U`, and `U†`

### Export Contract

Required exports after implementation:

- `quantum_singular_value_transformation.__all__` must include:
  - `QuantumSignalProcessing`
  - `ProjectorControlledPhase`
  - `QuantumSingularValueTransformation`
- `operators.oracle.__all__` must include:
  - `ProjectorControlledPhase`
  - `QuantumSingularValueTransformation`
- `operators.__all__` must include:
  - `ProjectorControlledPhase`
  - `QuantumSingularValueTransformation`

`tests/operators/test_package.py` must be updated accordingly.

### Minimal Working Example

```python
import numpy as np

from cqlib_algorithms.operators.oracle import (
    QuantumSingularValueTransformation,
    UnitaryDilation,
)

matrix = np.array(
    [
        [0.5, 0.0],
        [0.0, -0.25],
    ],
    dtype=np.complex128,
)

encoding = UnitaryDilation(matrix, validate=False)
qsvt = QuantumSingularValueTransformation.from_chebyshev_coefficients(
    encoding,
    [0.0, 0.75],
)
circuit = qsvt.build_circuit()
```

### Dependencies

No new project dependencies are required.

Allowed dependencies for this slice:

- `cqlib>=1.4.0b1,<1.5`
- `numpy>=2.1.0`

Must not introduce:

- `scipy`
- `pennylane`
- external phase-solving packages
- optimizer/training frameworks

### Deferred Items

The following are explicitly deferred beyond this specification:

- rectangular QSVT with alternating row/column projector dimensions
- non-standard block encodings with `Pi != PiTilde`
- public matrix-level reference assemblers as user-facing API
- generic projector definitions over arbitrary computational-basis prefixes
- phase-convention compatibility layers for external QSVT frameworks
