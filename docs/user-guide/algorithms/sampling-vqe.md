# 采样型变分量子本征求解器（Sampling VQE）

`SamplingVQE` 用有限-shot 计算基联合采样最小化对角 Hermitian 目标。当前目标只能由
$`I`$ 和 $`Z`$ Pauli 因子组成，因此每个比特串 $`x`$ 都有确定目标值 $`C(x)`$，一次
全寄存器计数分布即可计算普通样本均值

```math
\widehat{E}=\frac{1}{N}\sum_x n_xC(x)
```

## 联合采样语义

每个优化目标点只创建一个覆盖全部量子比特的 `SamplerRequest`。同一次 counts 同时
计算全部 Pauli-Z 项，既保留项间关联，也使 `shots` 明确表示该目标点的总采样次数。
实现不复用逐 Pauli 项拆分预算的通用采样 Estimator。

算法只向 `Optimizer` 提供普通样本均值，不实现 CVaR 或自定义非线性聚合。主 `seed`
为每个目标评估和最终采样派生独立子种子；相同主种子及相同优化调用轨迹可重现实验。
量子比特 0 对应结果整数的最低有效位。

## 结果解释

优化结束后，算法对完全绑定的最终线路执行一次独立采样。`SamplingVQEResult` 包含：

- `optimal_value`：优化器最终目标点的样本均值；
- `final_sample_value`：独立最终采样的均值，有限 shots 下可能与前者不同；
- `final_distribution`：独立最终 counts 归一化后的不可变分布；
- `best_measurement`：整个优化过程中目标采样见过的最佳比特串、目标值和当次出现概率；
- `shots`：每个目标点评估和最终采样各自使用的采样次数。

最佳测量与独立最终分布是两个不同概念，因此最佳比特串不要求出现在
`final_distribution`。跨目标点比较时先选目标值更低者；目标值相同则选当次出现概率
更高者；仍相同则按比特串稳定排序。

## 示例

```python
from cqlib.qis import PauliString

from cqlib_algorithms import SamplingVQE
from cqlib_algorithms.operators import (
    COBYLAOptimizer,
    RealAmplitudesAnsatz,
    StatevectorSampler,
)

algorithm = SamplingVQE(
    StatevectorSampler(),
    RealAmplitudesAnsatz(1, reps=0),
    COBYLAOptimizer(maxiter=20),
)
result = algorithm.run(
    PauliString.from_str("Z"),
    {"theta_0": 0.2},
    shots=4096,
    seed=7,
)
print(result.final_sample_value, result.best_measurement.bitstring)
```

`compute_minimum_eigenvalue(...)` 与 `run(...)` 使用相同的输入和计算流程，返回同一类型的 `SamplingVQEResult`。

该组件适合组合优化和其他计算基对角目标，不接受含 $`X`$ 或 $`Y`$ 的一般 Hamiltonian。
有限-shot 噪声会影响优化轨迹和最终估计，结果不保证达到全局最小值。
