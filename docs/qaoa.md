# QAOA

`QAOA` 组合既有的状态制备、成本演化、混合演化、期望值估计与经典优化组件，
用于最大化计算基对角的经典成本 Hamiltonian。算法层不重新实现线路模拟、期望值、
梯度、参数更新或参数绑定。

## 数学约定

对成本 Hamiltonian `H_C`、混合 Hamiltonian `H_M` 和层数 `p`，算法构造

```text
|gamma, beta> = U_M(beta[p-1]) U_C(gamma[p-1]) ...
                U_M(beta[0]) U_C(gamma[0]) |s>
U_C(gamma) = exp(-i gamma H_C)
U_M(beta)  = exp(-i beta H_M)
```

线路时间顺序是初态、第 0 层成本演化、第 0 层混合演化，依次直到第 `p-1` 层。
默认 `|s>` 是全量子比特均匀叠加态，默认 `H_M = sum_i X_i`。量子比特 0 是
计算基索引的最低有效位。

实现采用 Farhi、Goldstone 与 Gutmann 提出的交替成本/混合演化形式；自定义混合
Hamiltonian 和固定初态对应后续 alternating-operator 框架中的通用组合方式：

- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization
  Algorithm*, arXiv:1411.4028.
- S. Hadfield et al., *From the Quantum Approximate Optimization Algorithm to a
  Quantum Alternating Operator Ansatz*, arXiv:1709.03489.

## 接口与参数

```python
import math

from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms import QAOA
from cqlib_algorithms.operators import (
    GradientDescent,
    ParameterShiftGradient,
    StatevectorEstimator,
)

cost = Hamiltonian(2)
cost.add_term(PauliString.from_str("II"), 0.5)
cost.add_term(PauliString.from_str("ZZ"), -0.5)

algorithm = QAOA(
    StatevectorEstimator(),
    GradientDescent(ParameterShiftGradient(), learning_rate=0.2),
)
result = algorithm.compute_maximum(
    cost,
    {
        "qaoa_0_gamma": math.pi / 2,
        "qaoa_0_beta": math.pi / 8,
    },
)
```

`reps` 必须是非布尔正整数。参数严格按层交错排列：
`prefix_0_gamma, prefix_0_beta, ..., prefix_{p-1}_gamma,
prefix_{p-1}_beta`。初值必须精确覆盖线路中的活跃参数。

成本输入必须满足：

- 是至少作用于一个量子比特的非空 `Hamiltonian`；
- 所有有效系数有限且整体 Hermitian；
- 每个有效 Pauli 项只含 `I` 和 `Z`，因此计算基对角元素可解释为经典目标值。

自定义 `mixer_operator` 必须非空、系数有限、整体 Hermitian 且与成本同宽；它不要求
计算基对角。自定义 `initial_state`
必须实现 `StatePreparation`、与成本同宽，并生成没有活跃符号的固定线路。成本、
混合 Hamiltonian 和演化策略在进入底层组件时均采用独立快照；状态制备组件遵循
自身的确定性和独立线路契约。

成本和混合 Hamiltonian 的有限性、Hermitian 性、有效 Pauli 相位聚合与非空快照
统一复用 Ansatz 层的共享校验；QAOA 仅额外约束成本的正宽度和计算基对角性。

## 最大化与结果语义

公共优化器协议统一执行最小化，因此 `QAOA` 把 `-H_C` 交给既有 `VQE` 路径，
而成本演化线路仍使用原始 `H_C`。这两个符号约定不能混淆：

- `QAOAResult.optimal_value` 是原始成本期望值的最大化结果；
- `QAOAResult.optimizer_result.optimal_value` 是负成本目标值，数值上是前者的相反数；
- `optimal_parameters` 是不可变参数映射；
- `optimal_circuit` 是独立且完全绑定的线路；
- `vqe_result` 保留完整的被委托结果，避免复制或弱化其校验。

结果表示优化器的最终迭代点，不承诺找到全局经典最优解。`shots=None` 申请精确
期望值；正整数 shots 原样传给 Estimator，实际支持范围由注入的 Estimator 决定。

## 组件复用边界

- `UniformSuperposition` 负责默认初态；
- `QAOACostEvolution` 负责每层 `exp(-i gamma H_C)`；
- `QAOAMixerEvolution` 负责每层 `exp(-i beta H_M)`；
- `VQE` 负责 Estimator 请求、Optimizer 请求、结果校验和完全参数绑定；
- 注入的 Optimizer 继续拥有梯度组件及其更新规则。

当前接口面向由对角 Pauli Hamiltonian 表达的无约束经典成本。约束保持型混合器可
作为自定义 `mixer_operator` 提供，但可行初态、约束闭包和问题编码正确性由调用方
负责。算法不自动从图或布尔表达式生成成本 Hamiltonian，也不执行最终采样或解码。
