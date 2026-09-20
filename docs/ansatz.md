# Ansatz 与 QAOA 演化层组件

本文档描述 `cqlib_algorithms.operators.ansatz` 中已经实现的公共组件。实现及测试基于
Cqlib Core `develop` 提交 `c3abe91a4adf804b6fb3304fab6ec3dc24a4ad85` 的
Python 接口。所有组件均显式继承 `cqlib_algorithms.protocols.Ansatz`，每次构建均
返回独立的 `Circuit`。

## 公共约定

- `num_qubits` 必须是大于零的 Python `int`，`bool` 不作为整数接受。
- `reps` 必须是非负 Python `int`。`reps=0` 是合法退化配置。
- `prefix` 必须是字符串，用于为新建的符号参数命名。
- 默认纠缠拓扑为 `EntanglementTopology.linear()`。自定义拓扑的边界、自环和重复边
  由 Core 验证。
- TwoLocal 系列的参数顺序为“旋转层、量子比特、该量子比特上的旋转门位置”。
- 组件构造时复制或冻结输入配置，调用方后续修改原对象不会改变已经构造的组件。

## TwoLocal 系列

### `TwoLocalAnsatz`

```python
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate
from cqlib_algorithms.operators import TwoLocalAnsatz

ansatz = TwoLocalAnsatz(
    4,
    reps=2,
    rotation_gates=(StandardGate.RY, StandardGate.RZ),
    entanglement_gate=StandardGate.CX,
    entanglement=EntanglementTopology.circular(),
)
circuit = ansatz.build_circuit("theta")
```

它直接组合并委托 Core `TwoLocal`。旋转门只允许未绑定的 `RX`、`RY`、`RZ`，
纠缠门只允许未绑定的 `CX`、`CY`、`CZ`。每个 repetition 包含一个旋转层和一个
纠缠层；默认在末尾增加一个旋转层。参数数量为

```text
num_qubits * len(rotation_gates) *
(reps + (0 if skip_final_rotation_layer else 1))
```

空 `rotation_gates` 合法，此时线路可以只包含纠缠门。

### `RealAmplitudesAnsatz`

`RealAmplitudesAnsatz` 将 TwoLocal 固定为 `RY` 旋转与 `CX` 纠缠。所有固定参数下
的门矩阵均为实数。参数数量为
`num_qubits * (reps + (not skip_final_rotation_layer))`。

### `EfficientSU2Ansatz`

`EfficientSU2Ansatz` 将每个单比特参数块固定为依次执行 `RY`、`RZ`，纠缠门固定为
`CX`。参数数量为
`2 * num_qubits * (reps + (not skip_final_rotation_layer))`。

### `HardwareEfficientAnsatz`

`HardwareEfficientAnsatz` 是本库冻结的一种数字门模型实现：每个单比特参数块为
`RZ`、`RX`、`RZ` Euler 序列，之后按 Core 拓扑执行 `CX`、`CY` 或 `CZ` 纠缠。
参数数量为
`3 * num_qubits * (reps + (not skip_final_rotation_layer))`。

“硬件高效拟设”并没有唯一线路定义；这个组件不会自动查询设备、选择本征门或执行
拓扑映射。调用方应传入与目标设备相符的 `EntanglementTopology` 和纠缠门，后续仍需
经过编译和映射流程。

## `PauliTwoDesignAnsatz`

该组件先在每个量子比特上执行 `RY(pi/4)`，然后执行 `reps + 1` 个参数化旋转层。
每个参数槽的旋转轴在构造时从 `RX`、`RY`、`RZ` 中均匀抽样一次并冻结。前
`reps` 个旋转层之后依次执行偶边 `(0,1), (2,3), ...` 和奇边
`(1,2), (3,4), ...` 的 `CZ` brickwork 纠缠层。

```python
from cqlib_algorithms.operators import PauliTwoDesignAnsatz

ansatz = PauliTwoDesignAnsatz(4, reps=2, seed=7)
circuit = ansatz.build_circuit("theta")
```

参数顺序为“旋转层、量子比特”，数量为 `num_qubits * (reps + 1)`。相同的非空
`seed` 在本实现的不同实例间生成相同轴序列；同一实例重复构建始终确定。
`seed=None` 允许不同实例使用不同轴。这里不承诺与 Qiskit 或其他框架在相同 seed
下产生相同的随机轴序列。

## QAOA 单层演化

`QAOACostEvolution` 构造

```text
exp(-i * gamma * H_C)
```

并将唯一参数命名为 `{prefix}_gamma`。`QAOAMixerEvolution` 构造

```text
exp(-i * beta * H_M)
```

并将唯一参数命名为 `{prefix}_beta`。后者默认使用横场混合哈密顿量
`H_M = sum_i X_i`，也可接收同宽的自定义 Hermitian `Hamiltonian`。

```python
from cqlib.qis import Hamiltonian, PauliString
from cqlib_algorithms.operators import QAOACostEvolution, QAOAMixerEvolution

cost = Hamiltonian(2)
cost.add_term(PauliString.from_str("ZZ"), 0.5)

cost_layer = QAOACostEvolution(cost).build_circuit("qaoa")
mixer_layer = QAOAMixerEvolution(2).build_circuit("qaoa")
```

两个组件都委托 Core `PauliEvolutionAnsatz`，并对传入 Hamiltonian 做防御性复制。
默认 `EvolutionStrategy.auto(steps=1)`：对相互对易项使用精确乘积，对不对易项使用
一阶 Lie–Trotter。也可以显式传入 Core `EvolutionStrategy`。组件仅表示单个成本层
或混合层，不包含初态、层间参数管理和完整 QAOA 线路。

## 理论与接口参考

- Farhi、Goldstone、Gutmann，
  [A Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028)，
  给出交替成本与混合哈密顿量演化的 QAOA 定义。
- Kandala 等，
  [Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets](https://arxiv.org/abs/1704.05018)，
  说明按设备可用相互作用设计浅层变分线路的思想。
- Nakata 等，
  [Unitary 2-designs from random X- and Z-diagonal unitaries](https://arxiv.org/abs/1502.07514)，
  以及 McClean 等，
  [Barren plateaus in quantum neural network training landscapes](https://arxiv.org/abs/1803.11173)，
  是 Pauli two-design 类随机线路的理论和应用参考。
- IBM Quantum 官方接口文档：
  [TwoLocal](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.TwoLocal)、
  [RealAmplitudes](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.RealAmplitudes)、
  [EfficientSU2](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.EfficientSU2)、
  [PauliTwoDesign](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliTwoDesign)。

这些资料用于核对数学结构和接口组织；实现不依赖 Qiskit，也不主张跨框架的默认值、
随机序列或门顺序完全相同。
