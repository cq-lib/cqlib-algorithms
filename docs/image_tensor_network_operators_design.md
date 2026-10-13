# Image encoding and tensor-network operator design

## 1. Scope and baseline

- Target branch: `feat/operators-image-tensor-networks`.
- Baseline: `dev_fang` at `ea9edf1aaa6fb77aa4c959a5f22de93a8bd75f82`.
- Deliverables covered by this design: `ImageAngleEncoding`, the BRQI-backed
  `ImageBasisEncoding` (also exported as `BRQI`), `MPSAnsatz`, `TTNAnsatz`, and
  `MERAAnsatz`.
- The two image components implement the existing `StatePreparation` protocol.
  The three tensor-network components implement the existing `Ansatz` protocol
  through `BaseAnsatz`.
- No Qiskit or piQture runtime dependency is added. The implementations use only
  Cqlib, Python's standard library, and the project's existing NumPy test
  dependency.

This is a deliberately small first release. It provides deterministic circuit
components with explicit data layout and parameter order. It does not provide
image loading, resizing, colour-space conversion, train/test pipelines,
arbitrary tensor contraction, bond-dimension management, qubit reuse, or a
claim of quantum data-loading speedup.

## 2. Evidence and compatibility audit

### 2.1 Local contracts and Cqlib API

The design was checked against the following repository files:

- `src/cqlib_algorithms/protocols/state_preparation.py`;
- `src/cqlib_algorithms/protocols/ansatz.py`;
- `src/cqlib_algorithms/operators/ansatz/base_ansatz.py`;
- existing `AngleEncoding`, `BasisEncoding`, `SimplifiedTwoDesignAnsatz`, and
  their independent matrix tests;
- the Core compatibility range `cqlib>=1.4.0b1,<1.5` in `pyproject.toml`.

The required Core operations are already exercised in the repository:
`Circuit(num_qubits)`, `Circuit.ry(qubit, angle_or_parameter)`,
`Circuit.cx(control, target)`, `Circuit.x(qubit)`, `Circuit.validate()`,
`Circuit.assign_parameters(...)`, `Circuit.to_matrix()`, `Circuit.operations`,
and `Circuit.used_symbols`. No unverified Core builder is required.

All five components follow the frozen four-method protocol:

- `validate() -> None` checks the stored snapshot without modifying it;
- `build_circuit(prefix: str) -> Circuit` validates `prefix` and returns a new,
  independent circuit on every call;
- `num_parameters() -> int` returns logical feature/trainable slots;
- `num_qubits() -> int` returns the generated circuit width.

### 2.2 External references

The primary implementation reference is piQture 0.2.0, repository
`SaashaJoshi/piQture`, inspected at commit
`f475f66a4cc3bbed61e1e0bfcd4d9608a484c6f8`:

- image angle encoding:
  <https://github.com/SaashaJoshi/piQture/blob/f475f66a4cc3bbed61e1e0bfcd4d9608a484c6f8/piqture/embeddings/angle_encoding.py>;
- image input abstraction:
  <https://github.com/SaashaJoshi/piQture/blob/f475f66a4cc3bbed61e1e0bfcd4d9608a484c6f8/piqture/embeddings/image_embedding.py>;
- piQture BRQI implementation, used only as a negative compatibility audit:
  <https://github.com/SaashaJoshi/piQture/blob/f475f66a4cc3bbed61e1e0bfcd4d9608a484c6f8/piqture/embeddings/image_embeddings/brqi.py>;
- MPS, TTN, and MERA layouts:
  <https://github.com/SaashaJoshi/piQture/tree/f475f66a4cc3bbed61e1e0bfcd4d9608a484c6f8/piqture/tensor_networks>;
- simple two-qubit block:
  <https://github.com/SaashaJoshi/piQture/blob/f475f66a4cc3bbed61e1e0bfcd4d9608a484c6f8/piqture/gates/two_qubit_unitary.py>.

Scientific references used to fix the intended topology are:

- Grant et al., *Hierarchical quantum classifiers*, npj Quantum Information 4,
  65 (2018), <https://doi.org/10.1038/s41534-018-0116-9>;
- Vidal, *Class of Quantum Many-Body States That Can Be Efficiently Simulated*,
  Physical Review Letters 101, 110501 (2008),
  <https://doi.org/10.1103/PhysRevLett.101.110501>;
- Rieser, Koster, and Raulf, *Tensor networks for quantum machine learning*,
  Proceedings A 479, 20230218 (2023),
  <https://doi.org/10.1098/rspa.2023.0218>;
- Guala et al., *Practical overview of image classification with
  tensor-network quantum circuits*, Scientific Reports 13, 4427 (2023),
  <https://doi.org/10.1038/s41598-023-30258-y>.
- Li et al., *A Quantum Image Representation Based on Bitplanes*, IEEE Access
  6, 62396--62404 (2018),
  <https://doi.org/10.1109/ACCESS.2018.2871691>.

This design adopts the reference meanings, not the reference API verbatim.
piQture's angle encoder mixes fixed values and symbolic parameters without a
stable colour layout; its MERA parameter names contain a random UUID, and some
depth/count behaviour is not a stable public contract. Its BRQI implementation
creates position and eight colour qubits, does not condition each colour write
on the current pixel address, and appends measurements. It therefore does not
prepare the formal BRQI state and is not an implementation oracle for this
work. Cqlib uses the published BRQI definition, copied numerical input,
explicit register layout, deterministic names, and closed-form resource
counts. No source implementation is copied.

## 3. Shared conventions

### 3.1 Image data convention

The image components use channel-major flattened image input:

```python
image_shape: tuple[int, int]                 # (height, width)
channels: Sequence[Sequence[float | int]]   # channel-major, flattened rows
```

For height `H`, width `W`, and `C` channels, every channel contains exactly
`H * W` values in row-major order. Element `channels[c][r * W + x]` is the
value at channel `c`, row `r`, column `x`. A grayscale image has one channel;
RGB may be supplied to `ImageAngleEncoding` as three channels in
caller-selected R/G/B order. `ImageBasisEncoding` is the standard grayscale
BRQI and requires exactly one channel. Neither component performs colour-space
conversion.

`image_shape` and all channel containers are copied to immutable tuples during
construction. Strings and bytes are not accepted as sequences. Boolean values
are rejected where integers or real numbers are required.

### 3.2 Tensor-network block

The first release uses one real, two-parameter block throughout:

\[
B_{a,b}(\alpha,\beta)
=\operatorname{CX}_{a\rightarrow b}
  \left(R_Y^{(a)}(\alpha)\otimes R_Y^{(b)}(\beta)\right).
\]

Circuit time order is `RY(a, alpha)`, `RY(b, beta)`, then `CX(a, b)`. This is
the real simple block used by the hierarchical-classifier reference and the
piQture simple real parameterization. The first release intentionally excludes
general 6/15-parameter two-qubit blocks and auxiliary-qubit variants. Those are
separate future features, not flags hidden inside these five APIs.

All tensor-network parameters are allocated in gate-appearance order and named
`{prefix}_0`, `{prefix}_1`, ... . A block consumes the parameter for its first
qubit and then the parameter for its second qubit. No parameter is shared.

## 4. Public components

### 4.1 `ImageAngleEncoding`

Location: `operators/state_preparation/image_angle_encoding.py`.

```python
ImageAngleEncoding(
    image_shape: tuple[int, int],
    channels: Sequence[Sequence[float]],
    *,
    value_range: tuple[float, float] = (0.0, 255.0),
)
```

For `N = H * W`, scalar index `p = r * W + x`, and

\[
q(c,r,x)=cN+p,
\qquad
\theta_{c,r,x}=\pi\frac{v_{c,r,x}-v_{\min}}
                         {v_{\max}-v_{\min}},
\]

the circuit applies `RY(theta)` to qubit `q`. Starting from all zeroes, it
prepares the product state

\[
\bigotimes_{q=C N-1}^{0}
\left(\cos(\theta_q/2)|0\rangle+
      \sin(\theta_q/2)|1\rangle\right),
\]

where the displayed tensor order reflects Cqlib's little-endian matrix order.
Each pixel-channel scalar uses one qubit; this is not amplitude-compressed
image encoding and it has no position-address register.

Contract:

- `num_qubits()` is `C * H * W`;
- `num_parameters()` is `0`, because the input image is copied numerical data;
- `build_circuit(prefix)` checks that `prefix` is a string, otherwise ignores
  it, and applies one numerical `RY` per scalar in increasing qubit order;
- `validate()` requires positive integer `H` and `W`, at least one channel,
  exact channel lengths, finite real values, finite range endpoints,
  `v_min < v_max`, and every value in the inclusive range;
- wrong types raise `TypeError`; invalid dimensions, lengths, non-finite values,
  ranges, or out-of-range pixels raise `ValueError`.

The explicit range makes the scaling reproducible. Values at the lower and
upper endpoints map to `0` and `pi`; no clipping or implicit normalization is
performed.

### 4.2 `ImageBasisEncoding`

Location: `operators/state_preparation/image_basis_encoding.py`.

```python
ImageBasisEncoding(
    image_shape: tuple[int, int],
    channels: Sequence[Sequence[int]],
)

BRQI = ImageBasisEncoding
```

This component prepares the standard 8-bit grayscale bit-plane representation
of quantum images (BRQI). The public name `ImageBasisEncoding` is retained to
match the approved `image_basis` capability ID and the existing module path.
`BRQI` is an exact alias of the same class, not a subclass or a second
implementation.

Let `H = 2**h` and `W = 2**w` be the image height and width. The single input
channel contains integer intensities `v_(y,x)` in `[0, 255]`, where `y` is the
row and `x` is the column. The input container remains flattened in row-major
order at index `y*W+x`. Let the fixed bit-plane count be `d=8=2**3`.
Plane `l=0` is the least significant plane and

\[
g(y,x,l)=\left(\left\lfloor v_{y,x}/2^l\right\rfloor\bmod 2\right),
\qquad 0\leq l<8.
\]

The target state is

\[
|I_{\mathrm{BRQI}}\rangle
=\frac{1}{\sqrt{H W d}}
  \sum_{l=0}^{d-1}\sum_{y=0}^{H-1}\sum_{x=0}^{W-1}
  |g(y,x,l)\rangle_V |x\rangle_X |y\rangle_Y |l\rangle_L,
\qquad d=8.
\]

The labels in this equation identify logical registers. Cqlib's physical
little-endian mapping is frozen as follows:

- value register `V`: qubit `0`;
- column register `X`: qubits `1, ..., w`, with qubit `1` holding the least
  significant bit of `x`; this register is empty when `W=1`;
- row register `Y`: qubits `w+1, ..., w+h`, with qubit `w+1` holding the least
  significant bit of `y`; this register is empty when `H=1`;
- plane register `L`: qubits `w+h+1, ..., w+h+3`, with qubit `w+h+1` holding
  the least significant bit of `l`.

Consequently the Cqlib matrix-vector basis index of one summand is

\[
\operatorname{index}(y,x,l)
=g(y,x,l)+2x+2W y+2WH l,
\]

and the usual high-to-low-qubit display is
`|l>_L |y>_Y |x>_X |g>_V`. This exact mapping must be used by documentation
and tests; register order must not be inferred from textual tensor-factor
order alone.

#### Shape, bit depth, and channel scope

- `H` and `W` must each be powers of two with non-negative integer exponents
  (including 1). Thus every basis value of
  the row and column registers identifies a real pixel. There is no implicit
  padding, no invalid address, and no crop metadata.
- The first release is fixed to the standard grayscale depth of 8 bits. It
  therefore always uses three plane-index qubits. `bit_depth` is removed from
  the constructor. Eight is itself a power of two, so every plane-register
  address is valid; configurable depths are deferred because they would be a
  generalized, not standard, BRQI contract.
- Exactly one channel is accepted. The channel container is retained only to
  stay consistent with the image operator input convention. Zero or multiple
  channels raise `ValueError`.
- RGB BRQI is not emulated by concatenating three independent grayscale states.
  A future colour version requires a separate two-qubit channel register,
  normalization and unused-channel-address policy, and must receive a distinct
  reviewed contract.

#### Circuit construction

`build_circuit` returns an unmeasured state-preparation circuit:

1. Create `Circuit(h + w + 4)` in the register layout above.
2. Apply `H` to all `w` column, `h` row, and three plane qubits. This
   prepares the uniform address state
   `(1/sqrt(H*W*8)) sum_(l,y,x) |0>_V |x>_X |y>_Y |l>_L`.
3. For every `(y,x,l)` such that `g(y,x,l)=1`, append an `X` on value qubit
   `0` controlled on all column, row, and plane qubits matching that exact
   address.

Core multi-controlled gates only use closed controls. For one marked address,
the implementation must therefore:

1. collect controls in stable physical order `[1, ..., w+h+3]`, namely the
   little-endian bits of `x`, then `y`, then `l`;
2. apply `X` to every control whose corresponding bit in `(x,y,l)` is zero;
3. append
   `StandardGate.X.control(w + h + 3)` with
   `Circuit.append_mc_gate(..., [*controls, 0])`;
4. undo the open-control `X` gates in reverse order.

The zero-pattern conjugations turn the selected `(x,y,l)` address into the
all-one control state. They must surround each controlled write independently.
The implementation must not leave controls inverted, omit either pixel
coordinate, or apply one colour byte to all addresses. No measurement, reset,
classical condition, or sampler call belongs in this `StatePreparation`
component.

Contract:

- `num_qubits()` is `log2(H) + log2(W) + 4`;
- `num_parameters()` is `0`;
- `build_circuit(prefix)` checks the prefix, ignores its value, and returns a
  fresh, deterministic, unmeasured circuit;
- input is copied to immutable tuples before storage;
- values must be actual integers in `[0, 255]`; `bool` is rejected;
- `validate()` rechecks the immutable shape, one-channel rule, exact channel
  length, power-of-two dimensions, and grayscale range without mutation;
- wrong containers or value types raise `TypeError`; invalid dimensions,
  channel count/length, non-power-of-two height or width, or range raises
  `ValueError`.

This is a basis-valued, address-superposition image state. It is BRQI, not
NEQR or FRQI, and it does not imply QRAM access or efficient classical data
loading.

### 4.3 `MPSAnsatz`

Location: `operators/ansatz/mps_ansatz.py`.

```python
MPSAnsatz(num_qubits: int)
```

For `n >= 2`, the circuit applies the shared block once on each chain edge in
the exact time order

```text
(0, 1), (1, 2), ..., (n - 2, n - 1).
```

This nearest-neighbour sweep is an MPS-inspired variational circuit with a
two-qubit local block. It is not an API for importing arbitrary classical MPS
tensors or selecting a general bond dimension.

Contract:

- `num_qubits()` is `n`;
- `num_parameters()` is `2 * (n - 1)`;
- parameter order is edge order, then first/second qubit within each block;
- `num_qubits` must be an actual integer at least 2;
- `validate()` recomputes and checks the deterministic topology;
- repeated builds with the same prefix are equal but independent.

### 4.4 `TTNAnsatz`

Location: `operators/ansatz/ttn_ansatz.py`.

```python
TTNAnsatz(num_qubits: int)
```

Start with active qubits `[0, ..., n - 1]`. At each level, pair active qubits
from left to right, apply `B(left, right)`, and retain `right` for the next
level. If the level has an odd final qubit, carry it forward unchanged. Repeat
until one root remains, then append one trainable `RY` on the root. All qubits
remain in the physical circuit; "retain" defines later connectivity and does
not imply measurement, reset, or discard.

Examples:

```text
n=5
level 0: (0,1), (2,3); active -> [1,3,4]
level 1: (1,3);        active -> [3,4]
level 2: (3,4);        active -> [4]
root: RY on 4
```

Contract:

- `num_qubits()` is `n`;
- every non-root input participates in exactly one reducing block, so there
  are `n - 1` blocks;
- `num_parameters()` is `2 * (n - 1) + 1`;
- parameter order is level, then pair from left to right, then the two block
  parameters; the final parameter is the root `RY`;
- `num_qubits` must be an actual integer at least 2.

### 4.5 `MERAAnsatz`

Location: `operators/ansatz/mera_ansatz.py`.

```python
MERAAnsatz(num_qubits: int)
```

The first release implements a complete binary MERA-inspired unitary circuit,
so `num_qubits` must be a power of two and at least 2. For each active list
`[a_0, ..., a_{m-1}]`, apply:

1. disentangler blocks on boundaries
   `(a_1,a_2), (a_3,a_4), ..., (a_{m-3},a_{m-2})`;
2. isometry-position blocks on
   `(a_0,a_1), (a_2,a_3), ..., (a_{m-2},a_{m-1})`;
3. retain the right qubit of each isometry pair for the next scale.

When one root remains, append one trainable `RY` to it. Disentanglers always
precede isometry-position blocks at the same scale. Both use the shared unitary
block but have independent parameters. This is a circuit ansatz inspired by
binary MERA causal connectivity; it does not claim to expose arbitrary MERA
tensors or non-unitary isometries.

For `n = 2**L`, scale width `m` contributes `m/2 - 1` disentanglers and `m/2`
isometry-position blocks. Therefore

\[
N_B=\sum_{j=0}^{L-1}(n/2^j-1)=2n-2-L,
\qquad
N_\theta=2N_B+1=4n-3-2L.
\]

Contract:

- `num_qubits()` is `n`;
- `num_parameters()` is `4*n - 3 - 2*log2(n)` using exact integer arithmetic;
- parameter order is scale, all disentanglers left-to-right, all
  isometry-position blocks left-to-right, then root `RY`;
- non-power-of-two widths are rejected instead of silently creating an
  irregular network;
- no `layer_depth` argument is exposed in the first release because a partial
  coarse-graining policy and output-root semantics would require a separate,
  reviewed contract.

## 5. Private implementation sharing

Only narrow private helpers should be shared:

1. `operators/state_preparation/_image_data.py`
   - validate/copy `(height, width)`;
   - reject text-like channel containers;
   - copy channel-major data and verify exact `H * W` lengths;
   - provide only layout-neutral image validation used by both encoders.
   Numeric validation remains explicit for real-valued and integer-valued
   components so coercion cannot blur their different contracts.

The BRQI address-controlled write should remain a private function in
`image_basis_encoding.py`, because it is specific to the fixed value/column/
row/plane register layout. Do not broaden `_image_data.py` into a circuit
synthesis module and do not expose an address-oracle public abstraction.

2. `operators/ansatz/_tensor_network.py`
   - append the two-parameter real block and return the next parameter index;
   - generate immutable MPS edges, TTN levels, and MERA scale descriptors.

Do not add a public image base class, public tensor-network base class, gate
strategy protocol, or general unitary-block abstraction. The existing
`StatePreparation`, `Ansatz`, and `BaseAnsatz` contracts are sufficient.

## 6. Independent tests and acceptance criteria

### 6.1 Common protocol tests

For every component:

- `isinstance` matches its nominal protocol;
- `validate()` succeeds for valid construction and does not mutate state;
- `build_circuit` rejects a non-string prefix;
- `build_circuit(prefix).num_qubits == num_qubits()`;
- active symbol count/order agrees with `num_parameters()` for ansatze and is
  empty for fixed image encodings;
- two builds with the same prefix are equal, are not the same object, and
  mutating one does not affect the other.

### 6.2 Image-angle reference tests

- Independently compute each one-qubit vector
  `[cos(theta/2), sin(theta/2)]` and combine in reverse qubit order with NumPy
  Kronecker products; compare it to `Circuit.to_matrix()[:, 0]` with absolute
  tolerance `1e-12` and no global-phase relaxation.
- Use a two-channel `2 x 2` image with distinct values to detect channel and
  row-major transpositions.
- Cover lower endpoint, midpoint, upper endpoint, negative but permitted range,
  and multiple channels.
- Mutate every caller-owned list after construction and verify no change.
- Reject empty channels, ragged lengths, booleans, non-real values, NaN,
  infinity, invalid ranges, and out-of-range values.

### 6.3 Image-basis reference tests

- Build the complete expected BRQI state independently. For every `y`, `x`,
  and `l`, let `p=y*W+x`, then set amplitude `1/sqrt(8*H*W)` at index
  `((pixels[p] >> l) & 1) + 2*x + 2*W*y + 2*W*H*l`, with every other
  amplitude exactly zero. Compare against `Circuit.to_matrix()[:, 0]` at
  absolute tolerance `1e-12`, without global-phase relaxation.
- Test at least `1 x 1`, `1 x 2`, and `2 x 2` images with deliberately distinct
  bit patterns. These cases detect value/column/row/plane register swaps,
  reversed plane significance, row-major errors, missing coordinate controls,
  and leaked open controls.
- Independently assert normalization, exactly `8*H*W` nonzero amplitudes, equal
  positive phase on every valid address, and the correct decoded value bit for
  every `(y,x,l)` tuple.
- Assert the circuit contains no measurement, reset, or classical-control
  operation and has no active symbols. Check repeated builds and caller-input
  mutation independently.
- Inspect the high-level circuit for `log2(H)+log2(W)+3` Hadamards and one
  logical multi-controlled value write for each set image bit; full-state
  comparison, rather than the operation count, remains the correctness oracle.
- Reject zero/multiple channels, bools, negative values, `256`, non-integers,
  invalid shape, invalid channel length, and a non-power-of-two height or
  width. Include `1 x 1` to cover empty row/column registers.
- Assert `BRQI is ImageBasisEncoding` and both names are importable from the
  feature and aggregate operator namespaces.

### 6.4 Tensor-network topology and matrix tests

- Assert exact operation names, qubit targets, and `used_symbols` for:
  - MPS widths 2 and 4;
  - TTN widths 2, 3, and 5;
  - MERA widths 2, 4, and 8.
- Independently construct the dense matrix of the documented block from
  analytic `RY` and `CX` matrices and embed/multiply blocks in documented time
  order. Compare to a small bound Cqlib circuit at absolute tolerance `1e-12`.
  The reference must not call the implementation's topology or block helpers.
- Verify `U.conj().T @ U == I` independently for representative bound circuits.
- Check all closed-form parameter counts and that every declared parameter is
  active exactly once.
- Specifically assert MERA disentanglers precede isometry-position blocks at
  every scale and TTN's final root is qubit `n - 1`.
- Reject Boolean widths, widths below 2, and non-power-of-two MERA widths.

### 6.5 Repository acceptance

Implementation is complete only when all of the following hold:

- one public class per module, exported from the feature package and existing
  aggregate exports without introducing import cycles; `BRQI` is the sole
  permitted compatibility alias and points to `ImageBasisEncoding`;
- English type hints, docstrings, error messages, README/API documentation,
  and examples state the layout and non-compression limits;
- no new runtime dependency and no Qiskit/piQture import;
- relevant tests, `uv run --no-sync pytest`, strict configured mypy/Ruff checks,
  and `uv run --no-sync python -m build` pass in an environment with the
  approved Cqlib version;
- a candidate merge with the then-current `dev_fang` is regression-tested
  before the feature is marked safe to submit.

## 7. Known limitations and non-blocking decisions

- The names "angle/color encoding" and "bit-plane encoding" are ambiguous in
  the literature. `ImageAngleEncoding` remains a direct product encoding.
  `ImageBasisEncoding` specifically means the published 8-bit grayscale BRQI
  state with value, column, row, and plane registers; it must not be described
  as direct storage, FRQI, or NEQR.
- The tensor-network classes expose connectivity-based variational circuits,
  not arbitrary MPS/TTN/MERA tensor importers. "MPS", "TTN", and "MERA" must be
  documented as ansatz architecture names.
- The simple real block is sufficient for the requested L1 scope. General
  complex SU(4) blocks, bond-dimension controls, auxiliary qubits, partial
  MERA depth, and qubit reuse are deliberately deferred.
- No Core API gap blocks the five first-release components. The plan uses only
  already exercised circuit construction, parameters, validation, binding,
  matrix inspection, and multi-controlled standard-gate APIs.

## 8. BRQI resource bounds and migration

For `H=2**h`, `W=2**w`, and `d=8`, let

\[
s=\sum_{y=0}^{H-1}\sum_{x=0}^{W-1}\sum_{l=0}^{d-1}g(y,x,l)
\]

be the number of set image bits. BRQI uses exactly `h+w+4` logical qubits. The
high-level circuit has `h+w+3` Hadamards and `s` logical
`(h+w+3)`-controlled `X` writes. Open-control conjugation contributes at most
`2*s*(h+w+3)` ordinary `X` gates. Classical construction inspects `8HW` bits
and materializes a circuit linear in `s` times the address width. Core
decomposition of each multi-controlled gate may be much larger and depends on
the approved Core version and available ancilla policy; no small basis-gate
bound or hardware efficiency claim is made here.

The logarithmic logical-qubit count does not make classical image loading
efficient: general image preparation still enumerates all pixels and bit
planes, and can require `O(HW)` controlled writes. The component provides no
QRAM, decoder, measurement routine, image arithmetic, noise handling, or
loading-speedup guarantee. Dense state-matrix tests must remain small because
their vector dimension is `2**(h+w+4)=16HW`.

The earlier uncommitted direct-storage draft was migrated as one coherent
change:

1. `image_basis_encoding.py` now uses the BRQI registers and addressed
   controlled-write synthesis above, without stored/configurable `bit_depth`;
2. the class/module name is preserved, exact alias `BRQI` is exported, and
   exports are updated in
   `state_preparation/__init__.py`, `operators/__init__.py`, and package tests;
3. `_image_data.py` remains limited to shape/channel snapshot validation; the
   one-channel, power-of-two dimension, and `[0,255]` checks in the component;
4. `test_image_basis_encoding.py` uses independent
   complete-state BRQI tests and no-measurement assertions;
5. `docs/state_preparation.md` and README component summaries/examples use the
   BRQI contract rather than "direct storage", `C*d*H*W` qubits, or configurable
   bit depth;
6. `ImageAngleEncoding` and all MPS/TTN/MERA code and tests remain unchanged;
7. the image tests, full test suite, strict static checks, and build were rerun;
   the standard candidate-merge check against the current `dev_fang` baseline
   remains part of the independent audit phase.
