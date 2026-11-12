# 投影变分量子动力学（PVQD）

`PVQD` 用参数化线路近似封闭量子系统的实时演化。设当前拟设态为
<img src="../../assets/formulas/inline/inline-a98d9c7371d643b3.svg" alt="|\psi(\theta_n)\rangle" style="vertical-align: -4.000pt;">，每个时间步先构造短时目标态

<p align="center">
  <img src="../../assets/formulas/pvqd-target-state.svg" alt="|φn+1⟩=e-iΔtH|ψ(θn)⟩,">
</p>

再同时优化全部活跃参数，使投影损失

<p align="center">
  <img src="../../assets/formulas/pvqd-infidelity-loss.svg" alt="1-|⟨ψ(θn+1)|φn+1⟩|2">
</p>

最小。实现依据 Barison、Vicentini 和 Carleo 提出的投影变分动力学方法：
[An efficient quantum algorithm for the time evolution of parameterized circuits](https://doi.org/10.22331/q-2021-07-28-512)。

## 组件与边界

算法复用已有组件：

- `EvolvedOperatorAnsatz` 根据 `EvolutionStrategy` 构造每个短时演化；
- `ComputeUncomputeFidelity` 与注入的 `Sampler` 估计投影保真度；
- `Optimizer` 最小化每一步的不保真度；需要梯度时可使用现有梯度组件；
- 可选 `Estimator` 批量计算各时刻的辅助可观测量。

当前接口支持时间无关、有限且非空的 Hermitian `Hamiltonian` 或
`PauliString`。`fidelity_shots` 必须为正，因此投影目标是有限-shot 估计；`seed`
控制整条轨迹的可复现随机流。非对易 Hamiltonian 的线路误差由所选演化策略和分解
步数决定，不应与变分投影误差混为一谈。

每一步从上一步参数加上最近一次参数增量开始，并对全部活跃参数做全局优化。结果并不
保证经典优化器找到全局最优点，也不保证拟设能够表示精确演化态。

## 结果

`PVQDResult` 保存：

- `times` 和 `parameters`：包含初态在内的时间网格与参数轨迹；
- `fidelities`：每一步的投影保真度；
- `optimizer_results`：每一步的优化终止信息；
- `evolved_circuit`：最终时刻的完全绑定拟设线路；
- `observables`：可选的命名期望值历史。

`estimated_fidelity` 是全部步保真度的乘积，`estimated_error` 为一减去该乘积。二者是
投影质量诊断，不是最终态真实保真度的严格界。

## 示例

```python
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import PauliString

from cqlib_algorithms import PVQD
from cqlib_algorithms.operators import (
    COBYLAOptimizer,
    EfficientSU2Ansatz,
    StatevectorEstimator,
    StatevectorSampler,
)

ansatz = EfficientSU2Ansatz(1, reps=0)
initial = {name: 0.0 for name in ansatz.build_circuit("theta").used_symbols}
algorithm = PVQD(
    StatevectorSampler(),
    ansatz,
    COBYLAOptimizer(maxiter=200),
    estimator=StatevectorEstimator(),
    evolution_strategy=EvolutionStrategy.exact(),
)
result = algorithm.evolve(
    PauliString.from_str("X"),
    initial,
    0.5,
    num_timesteps=5,
    fidelity_shots=4096,
    auxiliary_operators={"z": PauliString.from_str("Z")},
    seed=7,
)

print(result.times)
print(result.fidelities)
print([entry.expectation for entry in result.observables["z"]])
```
