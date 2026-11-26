# QAOA

`QAOA` 组合既有的状态制备、成本演化、混合演化、期望值估计与经典优化组件，
通过 `run(cost, initial_point, ...)` 最小化或最大化计算基对角的成本 Hamiltonian。

## 数学约定

对成本 Hamiltonian $`H_C`$、一参数混合族 $`U_M(\beta)`$ 和层数 $`p`$，算法构造

```math
|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle=U_M(\beta_{p-1})U_C(\gamma_{p-1})\cdots U_M(\beta_0)U_C(\gamma_0)|s\rangle
```

其中 Hamiltonian Mixer 满足

```math
U_C(\gamma)=e^{-i\gamma H_C},\qquad U_M(\beta)=e^{-i\beta H_M}.
```

也可以用满足同一参数契约的 partitioned 或 controlled Mixer 线路替代
Hamiltonian Mixer。

线路时间顺序是初态、第 0 层成本演化、第 0 层混合演化，依次直到第 $`p-1`$ 层。
默认 $`\vert s\rangle`$ 是全量子比特均匀叠加态，默认 $`H_M=\sum_iX_i`$。量子比特 0 是
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
result = algorithm.run(
    cost,
    {
        "qaoa_0_gamma": math.pi / 2,
        "qaoa_0_beta": math.pi / 8,
    },
    objective_sense="maximize",
)
assert math.isclose(result.optimal_value, 1.0, abs_tol=1e-10)
```

`reps` 必须是非布尔正整数。参数严格按层交错排列：
`prefix_0_gamma, prefix_0_beta, ..., prefix_{p-1}_gamma, prefix_{p-1}_beta`。
初值必须精确覆盖线路中的活跃参数。

成本输入必须满足：

- 是至少作用于一个量子比特的非空 `Hamiltonian`；
- 所有有效系数有限且整体 Hermitian；
- 每个有效 Pauli 项只含 $`I`$ 和 $`Z`$，因此计算基对角元素可解释为经典目标值。

自定义 `mixer_operator` 必须非空、系数有限、整体 Hermitian 且与成本同宽；它不要求
计算基对角。也可通过 `mixer=Ansatz` 传入已经定义综合顺序的一参数 Mixer。线路 Mixer
必须与成本同宽、声明一个参数，并为第 `k` 层恰好生成 `prefix_k_beta`；它与
`mixer_operator` 互斥，且不能再指定 `mixer_strategy`。自定义 `initial_state`
必须实现 `StatePreparation`、与成本同宽，并生成没有活跃符号的固定线路。成本、
混合 Hamiltonian 和演化策略在进入底层组件时均采用独立快照；状态制备组件遵循
自身的确定性和独立线路契约。

成本和混合 Hamiltonian 的有限性、Hermitian 性、有效 Pauli 相位聚合与非空快照
统一复用 Ansatz 层的共享校验；QAOA 仅额外约束成本的正宽度和计算基对角性。

## 最小化、最大化与结果语义

`run` 默认最小化。最大化时传入原始收益 Hamiltonian 和 `objective_sense="maximize"`。两种方向的成本演化都使用原始 $`H_C`$。最大化时优化器最小化 $`-H_C`$ 的期望值，对外返回原始收益的期望值。

`compute_minimum_eigenvalue(...)` 和 `compute_maximum(...)` 分别对应上述两种方向，均返回 `VQAResult`。主要字段如下。

- `optimal_value`：原始目标在最终优化迭代点的期望值，而非历史最佳值；
- `objective_sense`：`"minimize"` 或 `"maximize"`，仅关键字参数；
- `optimizer_result`：原始优化器元数据；最小化时其目标值等于结果值，
  最大化时为结果值的相反数，构造时检查这一关系；
- `optimal_parameters`：按实际活跃符号顺序排列的参数映射；
- `optimal_circuit`：算法生成的完全绑定线路；
- `eigenvalue`：`optimal_value` 的兼容只读别名，不表示保证得到精确本征值。

`QAOAResult` 与 `VQEResult` 是 `VQAResult` 的导出别名，不是不同运行时类型。

`print(result)` 简要显示目标值、参数数量、优化方向、迭代次数和状态。结果不包含
checkpoint 或终态方差，优化器的最终迭代点也不承诺全局经典最优解。
`shots=None` 申请精确期望值；正整数 shots 原样传给
Estimator，实际支持范围由注入的 Estimator 决定。

## 构建线路与变体

`algorithm.build_circuit(cost_operator)` 只构建独立参数化线路，不执行优化：
`tuple(circuit.used_symbols)` 是初值需要覆盖的真实参数名与顺序。通常每层具有
gamma/beta 两个符号，但退化演化可能不包含某些逻辑参数槽，不应额外补造参数。

`QAOA` 不标记为 final；变体可覆盖该方法，优化入口会调用覆盖后的方法，再通过
公共请求契约仍会检查线路宽度、可观测量和精确参数绑定。变体负责自身线路的算法
语义，不会因此自动解除本版本计算基对角成本的限制。

## 组件复用边界

- `UniformSuperposition` 负责默认初态；
- `QAOACostEvolution` 负责每层 $`e^{-i\gamma H_C}`$；
- `QAOAMixerEvolution` 负责 Hamiltonian Mixer 的每层 $`e^{-i\beta H_M}`$；
- `operators.mixer` 的一参数组件负责受控与 partitioned Mixer 线路；
- `QAOA` 直接创建 Estimator/Optimizer 请求，校验返回结果并完成参数绑定；
- 注入的 Optimizer 继续拥有梯度组件及其更新规则。

当前接口面向由对角 Pauli Hamiltonian 表达的经典成本。约束保持型混合器可作为
自定义 `mixer_operator` 或线路 `mixer` 提供，但可行初态、约束闭包和问题编码正确性
由调用方负责。算法不自动从图或布尔表达式生成成本 Hamiltonian，也不执行最终采样
或解码。
