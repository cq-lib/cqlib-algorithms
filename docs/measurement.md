# Pauli measurement planning and grouping

The measurement package exposes three independent operators from
`cqlib_algorithms.operators.measurement`:

- `PauliGroupingStrategy(mode="none" | "qwc")` groups Pauli strings by a local
  tensor-product basis. Grouping is stable first-fit and returns tuples of
  original term indices. Scalar Pauli phases do not affect basis compatibility.
- `CommutingObservableGrouping` groups by full symplectic commutation. It is
  algebraic only. For example, XX and YY commute but are not qubit-wise
  commuting and cannot be passed to the local-basis planner as one group.
- `PauliMeasurementPlanner(circuit, terms, grouping="qwc")` returns a
  `MeasurementPlan` with immutable group metadata and independent, unmeasured
  basis circuits. `grouping="none"` creates one circuit per non-identity term.

`PauliString.get_pauli(q)` maps to `Circuit.qubits[q]` in the circuit's qubit
insertion order. Numeric qubit identifiers are not interpreted as positions.
The produced basis circuit rotates X with H and Y with Sdg followed by H. Z and
I require no rotation. It deliberately contains no terminal measurement: pass
it to `SamplerRequest` with `plan.qubits`, and let the Sampler append measurement
on a private execution circuit. This keeps measurement ownership consistent
with the public Sampler contract and preserves non-contiguous qubit IDs.

```python
from cqlib import Circuit
from cqlib.qis import PauliString
from cqlib_algorithms.operators.measurement import PauliMeasurementPlanner
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.sampler import SamplerRequest

preparation = Circuit(2)
preparation.h(0)
terms = [PauliString.from_str("IX"), PauliString.from_str("ZI")]
plan = PauliMeasurementPlanner(preparation, terms).plan()
for group_index, group in enumerate(plan.groups):
    basis_circuit = plan.build_circuit(group_index)
    request = SamplerRequest(basis_circuit, plan.qubits, 1000)
    result = StatevectorSampler().run([request])[0]
    print(group.term_indices, group.basis, result.counts)
```

The planner snapshots its Circuit and Pauli terms, including each term's
scalar phase, at construction. `MeasurementPlan.term(index)` returns a further
copy. An identity-only term has no readout group and is listed in
`identity_indices`; its phase and eventual Hamiltonian coefficient must be
handled by the estimator, not silently treated as `+1`. The planner neither
computes expectations nor chooses shots or mitigates results.

An input circuit must have at least one qubit, validate in Core and contain no
direct or nested measurement, reset or classical control. Barrier and Delay
operations are preserved as static no-op instructions. A composite Core
circuit gate is expanded for this check. All Pauli terms must be equal
width and match the circuit. Empty term lists are accepted and yield an empty
plan. Each `build_circuit` call makes a private Core circuit copy; the source
preparation and prior basis circuits are never mutated. The plan's group
indices are zero-based, with boolean, non-integer and out-of-range indices
rejected.

Scope decisions for the original operator inventory:

- Measurement layout optimization (46) is a compiler/device layout concern;
  Core already has layout and transpiler capabilities, so no duplicate
  measurement operator is implemented here.
- Measurement-error mitigation and readout calibration/model entries
  (47, 50, 51) require device-specific calibration data and an execution
  contract. This branch does not invent those inputs or provide placeholders.
- General commuting grouping (49) is provided as an algebraic partition only.
  Entangled joint-basis measurement synthesis is outside the local planner.

The Core Python-interface baseline inspected for this component is Cqlib
`develop` commit `c3abe91a4adf804b6fb3304fab6ec3dc24a4ad85`. The
implementation uses `PauliString.get_pauli`, `PauliString.commutes_with`,
`Circuit.qubits`, `Circuit.decompose`, `Circuit.__copy__`, `Circuit.h`,
and `Circuit.sdg` from that baseline. No Qiskit runtime dependency is
introduced.
