# 自适应变分量子本征求解器（ADAPT-VQE）

`AdaptVQE` 从给定 Hermitian 生成元池逐步构造拟设。对当前态

\[
g_j=i\langle\psi|[G_j,H]|\psi\rangle
\]

计算完整生成元池的梯度，选择绝对值最大的分量，将
`exp(-i theta_j G_j)` 追加到线路末尾，再用现有 `VQE` 重新优化全部活跃参数。
实现依据 Grimsley 等人提出的自适应变分方法：
[Nature Communications 10, 3007 (2019)](https://doi.org/10.1038/s41467-019-10988-2)。

## 组件与输入

- `Estimator` 同时计算对易子期望值和 VQE 目标；
- `Optimizer` 完成每轮全参数重优化；
- `operator_pool` 接收同宽、非空、非纯恒等的 Hermitian `Hamiltonian` 或
  `PauliString`；
- `initial_state` 是可选的固定零参数 `StatePreparation`，省略时使用全零态；
- 默认使用精确算符演化；显式传入 `EvolutionStrategy` 时按该策略综合，不把近似线路
  描述为精确演化。

输入算符和初态线路在构造时防御性快照。重复选择同一池元素是允许的，因为同一生成元
可能在不同线路位置再次有用。

## 停机与结果

主要停机条件是完整梯度向量的 L2 范数不超过 `gradient_threshold`。还支持：

- `eigenvalue_threshold`：候选新层带来的能量变化不超过阈值时回滚该候选层；
- `max_iterations`：限制已经接受的生成元数量；
- 重复后缀检测：阻止池选择序列在没有收敛时循环。

`AdaptVQEResult` 在标准变分结果之外保存已选择池索引、每次完整梯度范数、已接受态的
能量历史、最终最大梯度分量和明确的终止原因。若参考态已经满足梯度阈值，结果包含合法
的零参数线路，而不是构造虚假参数。

## 示例

```python
from cqlib.qis import PauliString

from cqlib_algorithms import AdaptVQE
from cqlib_algorithms.operators import (
    LBFGSBOptimizer,
    StatevectorEstimator,
    UniformSuperposition,
)

algorithm = AdaptVQE(
    StatevectorEstimator(),
    LBFGSBOptimizer(),
    [PauliString.from_str("Y")],
    initial_state=UniformSuperposition(1),
    gradient_threshold=1e-7,
)
result = algorithm.compute_minimum_eigenvalue(PauliString.from_str("Z"))

assert abs(result.eigenvalue + 1.0) < 1e-8
assert result.selected_operator_indices == (0,)
```

有限-shot 梯度是统计估计。使用 `shots` 时可通过 `seed` 指定整个运行的主种子；算法
会为参考能量、梯度和内部优化请求派生独立子种子。相同主种子、相同优化调用轨迹以及
遵守请求种子的 Estimator 可以复现运行。阈值停机只表示当前估计满足条件，不等价于
统计置信区间，也不保证生成元池能够表示精确基态。默认的精确算符演化在大规模、
多项且不对易的生成元上可能代价较高；此时应显式选择经过评估的近似演化策略。
