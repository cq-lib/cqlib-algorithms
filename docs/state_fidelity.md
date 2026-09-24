# Circuit state fidelity

`ComputeUncomputeFidelity` estimates the squared overlap of two pure states
prepared by Cqlib circuits. It is a concrete operator backed by the existing
synchronous batch `Sampler` contract; the package intentionally does not add a
state-fidelity protocol while there is only one implementation and no consumer
requires interchangeable implementations.

## Mathematical definition

For

```text
|psi_a> = U_a |0...0>
|psi_b> = U_b |0...0>,
```

the pure-state fidelity is

```text
F(|psi_a>, |psi_b>) = |<psi_a|psi_b>|^2
                     = |<0...0| U_a† U_b |0...0>|^2.
```

The operator therefore submits one compact circuit that applies `U_b` first
and `U_a†` second, then asks the injected sampler to measure every qubit. The
frequency of the all-zero bitstring is the fidelity estimate `p`. For more
than one shot, the returned variance is the unbiased sample estimate of the
variance of the sample mean,

```text
p * (1 - p) / (shots - 1).
```

With one shot, this variance cannot be estimated and is `None`.

## Interface and circuit rules

All public types are defined in
`cqlib_algorithms.operators.state_fidelity.compute_uncompute` and exported from
both `cqlib_algorithms.operators.state_fidelity` and
`cqlib_algorithms.operators`:

- `CircuitFidelityRequest` stores two circuits, a positive shot count, and
  separate exact parameter-binding mappings.
- `CircuitFidelityResult` stores the fidelity and estimator variance.
- `ComputeUncomputeFidelity` accepts a concrete `Sampler` at construction and
  evaluates a batch.

The request is shallowly frozen because Cqlib circuits are mutable. Binding
mappings are defensively copied and made immutable, but callers must not mutate
either circuit during the request lifetime. The operator revalidates each
request immediately before execution. Bindings must cover exactly the live
symbols of their corresponding circuit, and values must be finite real
numbers.

The circuits must have the same positive width. Their qubits are associated by
insertion position, not by numeric qubit identifier, and are mapped onto a
compact `0..n-1` execution circuit. This supports non-contiguous and different
source identifiers without mutating either source.

Both circuits must be static, unitary, fully bindable, decomposable, and
invertible by the compatible Cqlib Core. Measurements, reset, delay, classical
data or control, dynamic control flow, and operations for which Core cannot
construct an inverse are rejected before sampling. Barriers and global phase
are supported. A complete request batch is validated and transformed before a
single sampler call, so a late invalid request cannot partially submit an
earlier request.

## Example

```python
from cqlib import Circuit

from cqlib_algorithms.operators import (
    CircuitFidelityRequest,
    ComputeUncomputeFidelity,
    StatevectorSampler,
)

first = Circuit(1)
first.h(0)

same_state = Circuit(1)
same_state.h(0)
same_state.set_global_phase(0.4)

request = CircuitFidelityRequest(first, same_state, shots=256)
result = ComputeUncomputeFidelity(StatevectorSampler()).run([request])[0]

assert result.fidelity == 1.0
assert result.variance == 0.0
```

`StatevectorSampler` is useful for local execution. A backend implementation
can be injected through the same `Sampler` contract without changing the
fidelity operator.

## Scope and references

This operator estimates fidelity only for pure states represented by state-
preparation circuits. It does not implement Uhlmann fidelity for density
matrices, a mixed-state SWAP test, a local-subsystem fidelity objective, or
measurement-error mitigation. Numerical state and density-matrix fidelity
already belongs to Cqlib Core rather than this circuit-execution component.

The construction follows the compute-uncompute overlap used for quantum
feature-map kernels in Havlíček et al., *Supervised learning with
quantum-enhanced feature spaces*, Nature 567, 209-212 (2019),
[arXiv:1804.11326](https://arxiv.org/abs/1804.11326). The interface is designed
for Cqlib's circuit, sampler, result, and qubit-order semantics.
