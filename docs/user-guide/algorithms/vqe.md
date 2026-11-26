# 变分量子本征求解器（VQE）

`VQE` 是一个完整算法组件，用参数化量子态在拟设可达子空间中最小化 Hermitian
算子的期望值：

```math
E(\theta)=\langle\psi(\theta)|H|\psi(\theta)\rangle, \qquad |\psi(\theta)\rangle=A(\theta)|0\rangle.
```

根据变分原理，归一化试探态满足
$`E(\theta)\geq E_0`$，其中 $`E_0`$ 是 $`H`$ 的最小本征值。因此，优化结果是
拟设可达子空间中的能量上界；算法不保证所选拟设能够表示全局基态，也不保证所选
经典优化器找到全局最优点。

理论依据采用 Peruzzo 等人提出的 VQE：
[A variational eigenvalue solver on a photonic quantum processor](https://arxiv.org/abs/1304.3061)。
接口与实现均基于 Cqlib 自身的组件协议和类型体系。

## 组件组合

构造函数接收三个公共组件：

- `Ansatz` 构建参数化试探线路；
- `Estimator` 计算 Hermitian `Hamiltonian` 或 `PauliString` 的期望值；
- `Optimizer` 最小化 Estimator 目标。需要梯度的优化器自行持有 `Gradient`，VQE
  不重复实现或选择梯度规则。

`run(operator, initial_point, *, shots=None)` 用拟设生成参数化线路，将可观测量、
初值和可选采样次数交给优化器。优化器通过注入的 Estimator 评估能量，决定参数更新与停止条件。
VQE 将最终参数绑定到线路上，返回优化器在该点已求得的能量。`compute_minimum_eigenvalue`
与 `run` 接受相同的参数并返回同一种结果。

## 参数与结果语义

`initial_point` 必须显式提供，并且键集合必须精确等于
`ansatz.build_circuit(prefix).used_symbols`。训练顺序采用该列表的顺序；
`Ansatz.num_parameters()` 表示逻辑参数槽数量，退化配置中未出现在实际线路里的槽
不会被虚构为训练参数。无活跃符号时传入空映射，现有优化器会走
`OptimizerStatus.NO_PARAMETERS` 路径并完成一次目标评估。

`VQAResult` 是 VQE 与 QAOA 共用的结果类型，`VQEResult` 为同一类型的别名。
包含：

- `optimal_value`：优化器在最终迭代点独立计算的目标值；`eigenvalue` 保留为只读别名；
- `optimal_parameters`：按训练顺序保存的不可变参数映射；
- `optimal_circuit`：独立、完全绑定且不含活跃符号的线路；
- `optimizer_result`：完整的迭代次数、终止状态和最终点元数据。

- `objective_sense`：VQE 固定为 `"minimize"`，结果值必须与优化器值一致。

返回值不包含终态估计方差或 checkpoint。`repr(result)` 简要展示值、参数数量、
方向、迭代次数与状态。

## 示例

下面使用现有精确 Estimator、参数移位梯度、梯度下降优化器和实振幅拟设，求
单量子比特 $`Z`$ 的最小能量：

```python
from cqlib.qis import PauliString

from cqlib_algorithms import VQE
from cqlib_algorithms.operators import (
    GradientDescent,
    ParameterShiftGradient,
    RealAmplitudesAnsatz,
    StatevectorEstimator,
)

algorithm = VQE(
    estimator=StatevectorEstimator(),
    ansatz=RealAmplitudesAnsatz(1, reps=0),
    optimizer=GradientDescent(
        ParameterShiftGradient(),
        learning_rate=0.4,
        maxiter=80,
        tolerance=1e-10,
    ),
)
result = algorithm.run(
    PauliString.from_str("Z"),
    {"theta_0": 0.4},
)

assert abs(result.eigenvalue + 1.0) < 1e-9
assert result.optimal_circuit.used_symbols == []
```
