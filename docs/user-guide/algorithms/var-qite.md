# 变分虚时演化（VarQITE）

`VarQITE` 用参数化量子态近似归一化虚时演化

```math
|\psi(\tau)\rangle=\frac{e^{-\tau H}|\psi(0)\rangle}{\|e^{-\tau H}|\psi(0)\rangle\|}
```

虚时演化会相对抑制高能本征态，因此常用于从具有非零基态重叠的初态逐步逼近低能态。
本实现采用消除全局相位自由度后的 McLachlan 变分投影。在每个时刻计算相位规范
不变的投影量子几何度规

```math
g_{ij}=\mathrm{Re}\left[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right]
```

并求解

```math
g\dot{\theta}=-\frac{1}{2}\nabla_\theta\langle\psi(\theta)|H|\psi(\theta)\rangle
```

参数使用固定步长 Forward Euler 更新。该离散化简单且结果可解释，但不自适应调整
步长，也不提供积分截断误差估计。

## 组件与数值边界

算法复用 `NaturalGradient`，由其组合 `StatevectorQGT` 和
`ParameterShiftGradient`：

- QGT 度规始终由态矢量精确计算；
- `shots=None` 时，能量梯度和轨迹观测由支持精确模式的 `Estimator` 计算；
- 指定 `shots` 时，梯度、能量和辅助观测使用注入 Estimator 的有限-shot 语义，
  但 QGT 仍为精确态矢量度规；
- `regularization=0` 对奇异度规使用最小范数最小二乘解；正值则求解
  $`g+\lambda I`$ 的岭正则系统，其中 $`\lambda`$ 为 `regularization`。

拟设必须满足 `StatevectorQGT` 与 `ParameterShiftGradient` 的共同边界：活跃参数可出现
在支持的旋转、相位、受控旋转和 `U` 门的仿射表达式中。测量、重置、延迟、动态经典
控制、嵌套线路门、非线性参数表达式和未支持的参数化门会被拒绝。态矢量 QGT 的内存与
线路宽度呈指数增长，因此当前实现面向可精确生成态矢量的规模。

虚时总时长和步数必须为正。更小步长通常降低 Euler 离散误差，但增加 QGT 和梯度
计算次数。能量不保证在有限-shot 噪声、过大步长、正则化或表达能力不足的拟设上逐步
下降。

## 结果

`VarQITEResult` 的 `times`、`parameters` 和 `energies` 均包含初始点；命名
`observables` 也在同一时间网格上给出。`evolved_circuit` 是最终参数完全绑定后的独立
线路，`final_parameters` 和 `final_energy` 分别是轨迹末项的便捷视图。

## 示例

下面用单量子比特实振幅拟设在 Pauli-Z Hamiltonian 下演化，并同步记录 Pauli-X
期望值：

```python
from cqlib.qis import PauliString

from cqlib_algorithms import VarQITE
from cqlib_algorithms.operators import RealAmplitudesAnsatz, StatevectorEstimator

ansatz = RealAmplitudesAnsatz(1, reps=0)
algorithm = VarQITE(StatevectorEstimator(), ansatz)

result = algorithm.run(
    PauliString.from_str("Z"),
    {"theta_0": 0.4},
    0.2,
    num_timesteps=20,
    auxiliary_operators={"x": PauliString.from_str("X")},
    seed=7,
)

print(result.final_parameters)
print(result.final_energy.expectation)
print([item.expectation for item in result.observables["x"]])
```

这里的 `theta_0` 是该单比特拟设在默认前缀下的旋转角。一次 `run` 返回初态和 20 次更新，共 21 个时刻。`evolve(...)` 接受相同参数。

变分虚时演化框架见 McArdle 等人的
[Variational ansatz-based quantum simulation of imaginary time evolution](https://doi.org/10.1038/s41534-019-0187-2)；
本实现使用的全局相位修正与投影 QGT 形式见 Yuan 等人的
[Theory of variational quantum simulation](https://doi.org/10.22331/q-2019-10-07-191)。
